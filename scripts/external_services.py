#!/usr/bin/env python3
"""
scripts/external_services.py
Bring-your-own services: use the Postgres, Redis, ClickHouse, S3 storage, Langfuse or
LiteLLM you already run instead of the bundled containers - but only after checking
that this stack can reach them and has the read AND write access it needs. Anything
not configured, unreachable, or without write access is self-hosted as usual.

  python3 scripts/external_services.py resolve --platform cuda   # probe + decide (run_all.sh)
  python3 scripts/external_services.py check                     # probe only, human report
  python3 scripts/external_services.py gateway-register          # external LiteLLM: add routes
  python3 scripts/external_services.py gateway-unregister

Configured in .env (EXTERNAL_* keys, see .env.example). `resolve` prints `export` lines
on stdout for run_all.sh and notes on stderr ("↻" lines = automatic fallbacks). Probes
use only the Python standard library: the Postgres wire protocol (SCRAM-SHA-256 / MD5,
optional TLS), the Redis protocol (ACL users, TLS), ClickHouse's HTTP interface, signed
S3 requests (SigV4) and the Langfuse / LiteLLM HTTP APIs. Every probe object is removed.
"""

import argparse
import base64
import datetime
import hashlib
import hmac
import json
import os
import re
import secrets
import shlex
import socket
import ssl
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configure_model import parse_env_file  # noqa: E402
from init_env import ENV_PATH, set_env_values  # noqa: E402

PROBE = "llmops_access_probe"
TIMEOUT = 8


class ProbeError(Exception):
    """A failed check; `kind` is unreachable | auth | denied | error."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def note(msg: str) -> None:
    print(msg, file=sys.stderr)


def config() -> Dict[str, str]:
    values = parse_env_file(ENV_PATH) if os.path.exists(ENV_PATH) else {}
    values.update(os.environ)
    return values


def connect(host: str, port: int) -> socket.socket:
    try:
        return socket.create_connection((host, port), timeout=TIMEOUT)
    except OSError as e:
        raise ProbeError("unreachable", f"cannot connect to {host}:{port} ({e.strerror or e})")


def tls_wrap(sock: socket.socket, host: str, verify: bool) -> socket.socket:
    ctx = ssl.create_default_context()
    if not verify:  # sslmode=require semantics: encrypted, certificate not verified
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx.wrap_socket(sock, server_hostname=host)


# ------------------------------------------------------------------------------
# PostgreSQL (wire protocol v3)
# ------------------------------------------------------------------------------
class Postgres:
    def __init__(self, url: str, database: Optional[str] = None):
        u = urllib.parse.urlsplit(url)
        q = dict(urllib.parse.parse_qsl(u.query))
        self.host, self.port = u.hostname or "localhost", u.port or 5432
        self.user = urllib.parse.unquote(u.username or "postgres")
        self.password = urllib.parse.unquote(u.password or "")
        self.database = database or (u.path.lstrip("/") or self.user)
        self.schema = q.get("schema")
        self.sslmode = q.get("sslmode", "prefer")
        self.sock = connect(self.host, self.port)
        self._buf = b""
        self._handshake()

    # framing ---------------------------------------------------------------
    def _send(self, tag: bytes, payload: bytes) -> None:
        self.sock.sendall(tag + struct.pack("!I", len(payload) + 4) + payload)

    def _recv_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ProbeError("unreachable", "connection closed by the server")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _recv(self) -> Tuple[bytes, bytes]:
        tag = self._recv_exact(1)
        (length,) = struct.unpack("!I", self._recv_exact(4))
        return tag, self._recv_exact(length - 4)

    @staticmethod
    def _error(payload: bytes) -> Tuple[str, str]:
        fields = {}
        for part in payload.split(b"\0"):
            if part:
                fields[part[:1].decode()] = part[1:].decode(errors="replace")
        return fields.get("C", ""), fields.get("M", "")

    # startup + authentication ---------------------------------------------
    def _handshake(self) -> None:
        if self.sslmode != "disable":
            self.sock.sendall(struct.pack("!II", 8, 80877103))  # SSLRequest
            answer = self._recv_exact(1)
            if answer == b"S":
                self.sock = tls_wrap(self.sock, self.host, verify=self.sslmode in ("verify-ca", "verify-full"))
            elif self.sslmode in ("require", "verify-ca", "verify-full"):
                raise ProbeError("unreachable", "server does not support TLS (sslmode requires it)")
        params = b"user\0" + self.user.encode() + b"\0database\0" + self.database.encode() + b"\0\0"
        self.sock.sendall(struct.pack("!II", len(params) + 8, 196608) + params)
        nonce = base64.b64encode(secrets.token_bytes(18)).decode()
        client_first_bare = f"n=,r={nonce}"
        salted = auth_message = None
        while True:
            tag, payload = self._recv()
            if tag == b"E":
                code, msg = self._error(payload)
                kind = "auth" if code in ("28P01", "28000") else "denied" if code == "42501" else \
                    "missing" if code == "3D000" else "error"
                raise ProbeError(kind, f"{code} {msg}")
            if tag == b"R":
                (method,) = struct.unpack("!I", payload[:4])
                if method == 0:
                    continue  # AuthenticationOk
                if method == 3:  # cleartext
                    self._send(b"p", self.password.encode() + b"\0")
                elif method == 5:  # md5
                    inner = hashlib.md5((self.password + self.user).encode()).hexdigest()
                    digest = "md5" + hashlib.md5(inner.encode() + payload[4:8]).hexdigest()
                    self._send(b"p", digest.encode() + b"\0")
                elif method == 10:  # SASL: SCRAM-SHA-256
                    first = ("n,," + client_first_bare).encode()
                    self._send(b"p", b"SCRAM-SHA-256\0" + struct.pack("!I", len(first)) + first)
                elif method == 11:  # SASL continue
                    server_first = payload[4:].decode()
                    attrs = dict(kv.split("=", 1) for kv in server_first.split(","))
                    salted = hashlib.pbkdf2_hmac("sha256", self.password.encode(), base64.b64decode(attrs["s"]),
                                                 int(attrs["i"]))
                    final_bare = f"c=biws,r={attrs['r']}"
                    auth_message = f"{client_first_bare},{server_first},{final_bare}".encode()
                    client_key = hmac.new(salted, b"Client Key", hashlib.sha256).digest()
                    signature = hmac.new(hashlib.sha256(client_key).digest(), auth_message, hashlib.sha256).digest()
                    proof = base64.b64encode(bytes(a ^ b for a, b in zip(client_key, signature))).decode()
                    self._send(b"p", f"{final_bare},p={proof}".encode())
                elif method == 12:  # SASL final: verify the server signature
                    server_key = hmac.new(salted, b"Server Key", hashlib.sha256).digest()
                    expected = base64.b64encode(hmac.new(server_key, auth_message, hashlib.sha256).digest()).decode()
                    if payload[4:].decode().split("v=", 1)[-1] != expected:
                        raise ProbeError("auth", "server signature mismatch")
                else:
                    raise ProbeError("error", f"unsupported authentication method {method}")
            elif tag == b"Z":
                return  # ReadyForQuery

    def query(self, sql: str) -> List[List[Optional[str]]]:
        self._send(b"Q", sql.encode() + b"\0")
        rows, error = [], None
        while True:
            tag, payload = self._recv()
            if tag == b"D":
                (count,) = struct.unpack("!H", payload[:2])
                pos, row = 2, []
                for _ in range(count):
                    (size,) = struct.unpack("!i", payload[pos:pos + 4])
                    pos += 4
                    row.append(None if size < 0 else payload[pos:pos + size].decode(errors="replace"))
                    pos += max(size, 0)
                rows.append(row)
            elif tag == b"E":
                error = self._error(payload)
            elif tag == b"Z":
                if error:
                    code, msg = error
                    raise ProbeError("denied" if code == "42501" else "error", f"{code} {msg}")
                return rows

    def close(self) -> None:
        try:
            self._send(b"X", b"")
            self.sock.close()
        except OSError:
            pass


def probe_postgres(url: str, database: Optional[str] = None, connections: int = 0) -> str:
    """Read + DDL/DML write in the target schema, and room for `connections` more sessions."""
    pg = Postgres(url, database)
    try:
        if connections:
            free = int(pg.query("SELECT current_setting('max_connections')::int - "
                                "current_setting('superuser_reserved_connections')::int - "
                                "(SELECT count(*) FROM pg_stat_activity)")[0][0])
            if free < connections:
                raise ProbeError("denied", f"only {free} free connections, this stack opens up to {connections}")
        table = f"{PROBE}_{secrets.token_hex(4)}"
        qualified = f'"{pg.schema}".{table}' if pg.schema else table
        version = pg.query("SHOW server_version")[0][0]
        pg.query(f"CREATE TABLE {qualified} (id int primary key, note text)")
        try:
            pg.query(f"INSERT INTO {qualified} VALUES (1, 'probe')")
            if pg.query(f"SELECT note FROM {qualified} WHERE id = 1")[0][0] != "probe":
                raise ProbeError("error", "read back a different value")
        finally:
            pg.query(f"DROP TABLE IF EXISTS {qualified}")
        return f"Postgres {version} at {pg.host}:{pg.port}/{pg.database}"
    finally:
        pg.close()


def with_param(url: str, key: str, value: str) -> str:
    """Adds a query parameter unless the URL already sets it."""
    u = urllib.parse.urlsplit(url)
    params = dict(urllib.parse.parse_qsl(u.query))
    params.setdefault(key, value)
    return urllib.parse.urlunsplit((u.scheme, u.netloc, u.path, urllib.parse.urlencode(params), u.fragment))


def postgres_database_url(url: str, database: str) -> str:
    u = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((u.scheme, u.netloc, "/" + database, u.query, u.fragment))


def ensure_postgres_database(url: str, database: str) -> str:
    """URL of `database` on the server of `url`, created when missing and allowed."""
    try:
        Postgres(url, database).close()
    except ProbeError as e:
        if e.kind != "missing":
            raise
        admin = Postgres(url)
        try:
            admin.query(f'CREATE DATABASE "{database}"')
        except ProbeError as e2:
            raise ProbeError("denied", f"database '{database}' does not exist and cannot be created ({e2})")
        finally:
            admin.close()
    return postgres_database_url(url, database)


# ------------------------------------------------------------------------------
# Redis (RESP2)
# ------------------------------------------------------------------------------
class Redis:
    def __init__(self, url: str):
        u = urllib.parse.urlsplit(url)
        self.host, self.port = u.hostname or "localhost", u.port or 6379
        self.sock = connect(self.host, self.port)
        if u.scheme == "rediss":
            self.sock = tls_wrap(self.sock, self.host, verify=False)
        self.file = self.sock.makefile("rb")
        user, password = urllib.parse.unquote(u.username or ""), urllib.parse.unquote(u.password or "")
        if password:
            self.call(*(["AUTH", user, password] if user and user != "default" else ["AUTH", password]), kind="auth")
        db = u.path.lstrip("/")
        if db:
            self.call("SELECT", db)

    def call(self, *args, kind: str = "denied"):
        out = f"*{len(args)}\r\n".encode()
        for a in args:
            b = str(a).encode()
            out += f"${len(b)}\r\n".encode() + b + b"\r\n"
        self.sock.sendall(out)
        return self._read(kind)

    def _read(self, kind: str):
        line = self.file.readline()
        if not line:
            raise ProbeError("unreachable", "connection closed by the server")
        prefix, rest = line[:1], line[1:].rstrip(b"\r\n").decode(errors="replace")
        if prefix == b"-":
            denied = rest.startswith(("NOPERM", "NOAUTH", "WRONGPASS", "READONLY", "ERR AUTH"))
            raise ProbeError("auth" if rest.startswith(("NOAUTH", "WRONGPASS")) else kind if denied else "error", rest)
        if prefix in (b"+", b":"):
            return rest
        if prefix == b"$":
            n = int(rest)
            return None if n < 0 else self.file.read(n + 2)[:-2].decode(errors="replace")
        if prefix == b"*":
            return [self._read(kind) for _ in range(max(int(rest), 0))]
        raise ProbeError("error", f"unexpected reply {line[:40]!r}")

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def probe_redis(url: str) -> str:
    r = Redis(url)
    try:
        r.call("PING")
        key = f"{PROBE}:{secrets.token_hex(4)}"
        try:
            r.call("SET", key, "1", "EX", "60")
            if r.call("GET", key) != "1":
                raise ProbeError("error", "read back a different value")
            r.call("INCRBY", key, "1")
            r.call("HSET", key + ":h", "f", "v")
            r.call("ZADD", key + ":z", "1", "m")
            r.call("LPUSH", key + ":l", "x")
            r.call("EVAL", "return redis.call('GET', KEYS[1])", "1", key)  # Lua: LiteLLM limits, Langfuse queue
        finally:
            for suffix in ("", ":h", ":z", ":l"):
                try:
                    r.call("DEL", key + suffix)
                except ProbeError:
                    pass
        info = r.call("INFO", "server") or ""
        version = re.search(r"redis_version:(\S+)", info)
        return f"Redis {version.group(1) if version else '?'} at {r.host}:{r.port}"
    finally:
        r.close()


# ------------------------------------------------------------------------------
# ClickHouse (HTTP interface)
# ------------------------------------------------------------------------------
def clickhouse(url: str, user: str, password: str, sql: str, database: str) -> str:
    query = urllib.parse.urlencode({"database": database}) if database else ""
    req = urllib.request.Request(f"{url.rstrip('/')}/?{query}", data=sql.encode(), method="POST",
                                 headers={"X-ClickHouse-User": user, "X-ClickHouse-Key": password})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=ssl.create_default_context()) as resp:
            return resp.read().decode(errors="replace").strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        code = re.search(r"Code: (\d+)", body)
        code = int(code.group(1)) if code else 0
        kind = "auth" if code in (192, 193, 516) or e.code == 401 else \
            "denied" if code in (164, 497) or "privileges" in body or "readonly" in body.lower() else "error"
        raise ProbeError(kind, body.strip().splitlines()[0][:200] if body.strip() else f"HTTP {e.code}")
    except (urllib.error.URLError, OSError) as e:
        raise ProbeError("unreachable", f"cannot reach {url} ({getattr(e, 'reason', e)})")


def clickhouse_migration_url(http_url: str, explicit: str) -> str:
    """Native-protocol URL Langfuse migrates over: EXTERNAL_CLICKHOUSE_MIGRATION_URL, else the
    standard native port that pairs with the HTTP one (8123 -> 9000, 8443 -> 9440)."""
    if explicit:
        return explicit
    u = urllib.parse.urlsplit(http_url)
    secure = u.scheme == "https"
    return f"clickhouse://{u.hostname}:{9440 if secure else 9000}" + ("?secure=true" if secure else "")


def probe_clickhouse(url: str, user: str, password: str, database: str, migration_url: str) -> str:
    version = clickhouse(url, user, password, "SELECT version()", database)
    table = f"{PROBE}_{secrets.token_hex(4)}"
    clickhouse(url, user, password, f"CREATE TABLE {table} (id UInt8) ENGINE = MergeTree ORDER BY id", database)
    try:
        clickhouse(url, user, password, f"INSERT INTO {table} VALUES (1)", database)
        if clickhouse(url, user, password, f"SELECT count() FROM {table}", database) != "1":
            raise ProbeError("error", "read back a different value")
    finally:
        clickhouse(url, user, password, f"DROP TABLE IF EXISTS {table}", database)
    # Langfuse also reads system tables (parts, mutations, tables) for housekeeping and deletes
    for system_table in ("parts", "mutations", "tables"):
        clickhouse(url, user, password, f"SELECT count() FROM system.{system_table} WHERE database = currentDatabase()", database)
    m = urllib.parse.urlsplit(migration_url)  # Langfuse migrates over the native protocol
    try:
        connect(m.hostname or "", m.port or 9000).close()
    except ProbeError as e:
        raise ProbeError("unreachable", f"native port for Langfuse migrations: {e} - set EXTERNAL_CLICKHOUSE_MIGRATION_URL")
    return f"ClickHouse {version} at {urllib.parse.urlsplit(url).netloc}/{database or 'default'}"


# ------------------------------------------------------------------------------
# S3-compatible object storage (SigV4)
# ------------------------------------------------------------------------------
def s3_request(method: str, endpoint: str, bucket: str, key: str, region: str, access: str, secret: str,
               path_style: bool, body: bytes = b"") -> bytes:
    u = urllib.parse.urlsplit(endpoint)
    host = u.netloc if path_style else f"{bucket}.{u.netloc}"
    path = f"/{bucket}/{urllib.parse.quote(key)}" if path_style else f"/{urllib.parse.quote(key)}"
    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(body).hexdigest()
    headers = {"host": host, "x-amz-content-sha256": payload_hash, "x-amz-date": amz_date}
    signed = ";".join(sorted(headers))
    canonical = "\n".join([method, path, "", *(f"{k}:{headers[k]}" for k in sorted(headers)), "", signed, payload_hash])
    scope = f"{day}/{region}/s3/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    k = ("AWS4" + secret).encode()
    for part in (day, region, "s3", "aws4_request"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    headers["authorization"] = (f"AWS4-HMAC-SHA256 Credential={access}/{scope}, SignedHeaders={signed}, "
                                f"Signature={signature}")
    req = urllib.request.Request(f"{u.scheme}://{host}{path}", data=body if method == "PUT" else None,
                                 method=method, headers={k: v for k, v in headers.items() if k != "host"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=ssl.create_default_context()) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="replace")
        code = re.search(r"<Code>([^<]+)</Code>", body_text)
        code = code.group(1) if code else f"HTTP {e.code}"
        kind = "auth" if code in ("InvalidAccessKeyId", "SignatureDoesNotMatch") else \
            "denied" if code == "AccessDenied" or e.code == 403 else "missing" if code == "NoSuchBucket" else "error"
        raise ProbeError(kind, code)
    except (urllib.error.URLError, OSError) as e:
        raise ProbeError("unreachable", f"cannot reach {endpoint} ({getattr(e, 'reason', e)})")


def probe_s3(endpoint: str, bucket: str, region: str, access: str, secret: str, path_style: bool) -> str:
    key = f"{PROBE}/{secrets.token_hex(6)}"
    args = (endpoint, bucket, key, region, access, secret, path_style)
    s3_request("PUT", *args, body=b"probe")
    try:
        if s3_request("GET", *args) != b"probe":
            raise ProbeError("error", "read back a different object")
    finally:
        s3_request("DELETE", *args)
    return f"S3 bucket '{bucket}' at {urllib.parse.urlsplit(endpoint).netloc}"


# ------------------------------------------------------------------------------
# HTTP APIs: Langfuse, LiteLLM
# ------------------------------------------------------------------------------
def http(method: str, url: str, headers: Optional[Dict[str, str]] = None, body=None, timeout: float = 20) -> Tuple[int, object]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as resp:
            raw = resp.read().decode(errors="replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        raw, status = e.read().decode(errors="replace"), e.code
    except (urllib.error.URLError, OSError) as e:
        raise ProbeError("unreachable", f"cannot reach {url.split('?')[0]} ({getattr(e, 'reason', e)})")
    try:
        return status, json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return status, raw


def probe_langfuse(url: str, public: str, secret_key: str) -> str:
    url = url.rstrip("/")
    status, health = http("GET", f"{url}/api/public/health", timeout=TIMEOUT)
    if status != 200:
        raise ProbeError("unreachable", f"health check returned HTTP {status}")
    if not isinstance(health, dict) or "status" not in health:
        raise ProbeError("error", "no Langfuse API at this URL")
    auth = {"Authorization": "Basic " + base64.b64encode(f"{public}:{secret_key}".encode()).decode()}
    status, projects = http("GET", f"{url}/api/public/projects", auth)
    if status in (401, 403):
        raise ProbeError("auth", "the project API keys were rejected")
    if status != 200 or not isinstance(projects, dict) or not isinstance(projects.get("data"), list):
        raise ProbeError("error", f"project lookup returned HTTP {status}: {str(projects)[:120]}")
    # write: one OpenTelemetry span - exactly how the gateway's langfuse_otel callback sends traces
    trace_id, now = uuid.uuid4().hex, int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e9)
    span = {"traceId": trace_id, "spanId": secrets.token_hex(8), "name": PROBE, "kind": 1,
            "startTimeUnixNano": str(now), "endTimeUnixNano": str(now + 1_000_000)}
    export = {"resourceSpans": [{"resource": {"attributes": [{"key": "service.name", "value": {"stringValue": PROBE}}]},
                                 "scopeSpans": [{"scope": {"name": PROBE}, "spans": [span]}]}]}
    status, result = http("POST", f"{url}/api/public/otel/v1/traces", auth, export)
    if status in (401, 403):
        raise ProbeError("denied", f"trace export rejected: {str(result)[:160]}")
    if status in (404, 405):
        raise ProbeError("error", "no OpenTelemetry endpoint (/api/public/otel needs Langfuse v3+)")
    if status >= 300:
        raise ProbeError("error", f"trace export returned HTTP {status}: {str(result)[:120]}")
    http("DELETE", f"{url}/api/public/traces/{trace_id}", auth)  # best effort cleanup
    name = (projects["data"] or [{}])[0].get("name", "?")
    return f"Langfuse at {urllib.parse.urlsplit(url).netloc} (project '{name}')"


def probe_litellm(url: str, master_key: str) -> str:
    url = url.rstrip("/")
    status, _ = http("GET", f"{url}/health/liveliness", timeout=TIMEOUT)
    if status != 200:
        raise ProbeError("unreachable", f"liveliness returned HTTP {status}")
    auth = {"Authorization": f"Bearer {master_key}"}
    status, keys = http("GET", f"{url}/key/list?size=1", auth)
    if status in (401, 403):
        raise ProbeError("auth", "the key was rejected (an admin / master key is needed)")
    if status != 200 or not isinstance(keys, dict):
        raise ProbeError("error", f"no LiteLLM admin API at this URL (key listing: HTTP {status})")
    status, created = http("POST", f"{url}/model/new", auth, {
        "model_name": PROBE, "litellm_params": {"model": "openai/probe", "api_base": "http://127.0.0.1:9", "api_key": "x"},
        "model_info": {"llmops_managed": True}})
    if status != 200:
        raise ProbeError("denied", "cannot add models (needs an admin key and STORE_MODEL_IN_DB=True on that proxy): "
                         + str(created)[:120])
    model_id = (created.get("model_id") or created.get("model_info", {}).get("id")) if isinstance(created, dict) else None
    remove_models(url, master_key, names={PROBE}, ids={model_id} if model_id else set())
    status, key = http("POST", f"{url}/key/generate", auth, {"key_alias": f"{PROBE}-{secrets.token_hex(3)}", "duration": "5m"})
    if status != 200 or not isinstance(key, dict) or not key.get("key"):
        raise ProbeError("denied", f"cannot issue virtual keys: {str(key)[:120]}")
    http("POST", f"{url}/key/delete", auth, {"keys": [key["key"]]})
    return f"LiteLLM at {urllib.parse.urlsplit(url).netloc} (admin, can add models and keys)"


def remove_models(url: str, master_key: str, names=frozenset(), ids=frozenset(), managed: str = "") -> int:
    """Deletes models by name / id, or the ones a stack registered for upstream `managed`."""
    auth = {"Authorization": f"Bearer {master_key}"}
    status, listing = http("GET", f"{url.rstrip('/')}/model/info", auth)
    removed = 0
    for m in (listing.get("data") or []) if status == 200 and isinstance(listing, dict) else []:
        info = m.get("model_info") or {}
        if m.get("model_name") in names or info.get("id") in ids or (managed and info.get("llmops_managed") == managed):
            http("POST", f"{url.rstrip('/')}/model/delete", auth, {"id": info.get("id")})
            removed += 1
    return removed


# ------------------------------------------------------------------------------
# Decision
# ------------------------------------------------------------------------------
def container_url(url: str) -> str:
    """Containers reach a service on this host through host.docker.internal, not localhost."""
    u = urllib.parse.urlsplit(url)
    if u.hostname in ("localhost", "127.0.0.1", "::1"):
        netloc = u.netloc.replace(u.hostname, "host.docker.internal", 1)
        return urllib.parse.urlunsplit((u.scheme, netloc, u.path, u.query, u.fragment))
    return url


def loopback_only(url: str) -> bool:
    """A service on this host that listens on loopback only is invisible to containers."""
    u = urllib.parse.urlsplit(url)
    if u.hostname not in ("localhost", "127.0.0.1", "::1"):
        return False
    try:
        bridge = subprocess_out(["docker", "network", "inspect", "bridge", "-f", "{{(index .IPAM.Config 0).Gateway}}"])
        socket.create_connection((bridge or "172.17.0.1", u.port or 0), timeout=3).close()
        return False
    except OSError:
        return True


def subprocess_out(cmd: List[str]) -> str:
    import subprocess
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def why(e: Exception) -> str:
    if not isinstance(e, ProbeError):  # a protocol surprise (timeout, TLS, odd reply) is never fatal
        return f"check failed ({type(e).__name__}: {e})"
    return {"unreachable": "unreachable", "auth": "credentials rejected", "denied": "write access denied",
            "missing": "not found"}.get(e.kind, "check failed") + f" ({e})"


class Resolver:
    def __init__(self, cfg: Dict[str, str], persist: bool = True):
        self.cfg = cfg
        self.persist = persist  # False for `check`: probe only, change nothing
        self.exports: Dict[str, str] = {}
        self.external: Dict[str, str] = {}   # service -> description
        self.selfhost: List[str] = []
        self.failed: Dict[str, str] = {}     # service -> ProbeError kind

    def get(self, key: str) -> str:
        return (self.cfg.get(key) or "").strip()

    def attempt(self, service: str, configured: bool, fn, *args) -> bool:
        if not configured:
            return False
        try:
            self.external[service] = fn(*args)
            note(f"  ✓ {service:<10}: external - {self.external[service]} (read/write OK)")
            return True
        except Exception as e:
            self.failed[service] = e.kind if isinstance(e, ProbeError) else "error"
            note(f"  ↻ {service}: external service {why(e)}: self-hosting it instead")
            return False

    def resolve(self) -> None:
        g = self.get
        # Langfuse (replaces bundled Langfuse + ClickHouse + MinIO when usable)
        langfuse_ext = self.attempt("Langfuse", bool(g("EXTERNAL_LANGFUSE_URL")), probe_langfuse,
                                    g("EXTERNAL_LANGFUSE_URL"), g("EXTERNAL_LANGFUSE_PUBLIC_KEY"), g("EXTERNAL_LANGFUSE_SECRET_KEY"))
        # LiteLLM: the final check (a request through it reaches our model) runs after the engine is up
        litellm_ext = False
        if g("EXTERNAL_LITELLM_URL") and not g("EXTERNAL_LITELLM_UPSTREAM_URL"):
            note("  ↻ LiteLLM: EXTERNAL_LITELLM_UPSTREAM_URL is not set (how that proxy reaches this host's router): "
                 "self-hosting the gateway instead")
        else:
            litellm_ext = self.attempt("LiteLLM", bool(g("EXTERNAL_LITELLM_URL")), probe_litellm,
                                       g("EXTERNAL_LITELLM_URL"), g("EXTERNAL_LITELLM_MASTER_KEY"))

        need_pg = {"litellm": not litellm_ext, "langfuse": not langfuse_ext}
        need_redis = not litellm_ext or not langfuse_ext
        unused = [name for name, given, needed in (
            ("Postgres", g("EXTERNAL_POSTGRES_URL"), any(need_pg.values())), ("Redis", g("EXTERNAL_REDIS_URL"), need_redis),
            ("ClickHouse", g("EXTERNAL_CLICKHOUSE_URL"), not langfuse_ext), ("S3", g("EXTERNAL_S3_BUCKET"), not langfuse_ext))
            if given and not needed]
        if unused:
            note(f"  • {', '.join(unused)}: not needed this run (your Langfuse / LiteLLM keep their own data)")

        # Postgres: gateway database = the URL's database; Langfuse needs a database of its own.
        # Connections this stack opens: gateway workers x 5 (pool limit) + Langfuse 2 x 10.
        pg_url = g("EXTERNAL_POSTGRES_URL")
        pg_for = {"litellm": False, "langfuse": False}
        workers = int(g("LITELLM_NUM_WORKERS")) if g("LITELLM_NUM_WORKERS").isdigit() else 4
        need_conn = (workers * 5 + 2 if need_pg["litellm"] else 0) + (20 if need_pg["langfuse"] else 0)
        if pg_url and any(need_pg.values()) and self.usable_from_containers("Postgres", pg_url):
            if need_pg["litellm"] and self.attempt("Postgres", True, probe_postgres, pg_url, None, need_conn):
                pg_for["litellm"] = True
                self.exports["LITELLM_DATABASE_URL"] = container_url(pg_url)
            if need_pg["langfuse"] and self.failed.get("Postgres") not in ("unreachable", "auth"):
                lf_url = g("EXTERNAL_LANGFUSE_POSTGRES_URL")
                try:
                    lf_url = lf_url or ensure_postgres_database(pg_url, "langfuse")
                    detail = probe_postgres(lf_url, None, 0 if pg_for["litellm"] else need_conn)
                    pg_for["langfuse"] = True
                    self.exports["LANGFUSE_DATABASE_URL"] = with_param(container_url(lf_url), "connection_limit", "10")
                    note(f"  ✓ {'Postgres':<10}: external for Langfuse - {detail}")
                except Exception as e:
                    note(f"  ↻ Postgres (Langfuse database): external {why(e)}: self-hosting it instead")
        if (need_pg["litellm"] and not pg_for["litellm"]) or (need_pg["langfuse"] and not pg_for["langfuse"]):
            self.selfhost.append("postgres")

        # Redis: shared by gateway cache / limits and the Langfuse queue
        redis_url = g("EXTERNAL_REDIS_URL")
        if need_redis:
            if redis_url and self.usable_from_containers("Redis", redis_url) and \
                    self.attempt("Redis", True, probe_redis, redis_url):
                self.exports["LITELLM_REDIS_URL"] = self.exports["LANGFUSE_REDIS_CONNECTION_STRING"] = container_url(redis_url)
            else:
                self.selfhost.append("redis")

        if not langfuse_ext:
            self.selfhost += ["langfuse", "langfuse-worker"]
            ch_url = g("EXTERNAL_CLICKHOUSE_URL")
            mig = clickhouse_migration_url(ch_url, g("EXTERNAL_CLICKHOUSE_MIGRATION_URL")) if ch_url else ""
            if ch_url and self.usable_from_containers("ClickHouse", ch_url) and self.attempt(
                    "ClickHouse", True, probe_clickhouse, ch_url, g("EXTERNAL_CLICKHOUSE_USER") or "default",
                    g("EXTERNAL_CLICKHOUSE_PASSWORD"), g("EXTERNAL_CLICKHOUSE_DB"), mig):
                self.exports.update({
                    "LANGFUSE_CLICKHOUSE_URL": container_url(ch_url), "LANGFUSE_CLICKHOUSE_MIGRATION_URL": container_url(mig),
                    "LANGFUSE_CLICKHOUSE_USER": g("EXTERNAL_CLICKHOUSE_USER") or "default",
                    "LANGFUSE_CLICKHOUSE_PASSWORD": g("EXTERNAL_CLICKHOUSE_PASSWORD"),
                    "LANGFUSE_CLICKHOUSE_DB": g("EXTERNAL_CLICKHOUSE_DB") or "default"})
            else:
                self.selfhost.append("clickhouse")
            region = g("EXTERNAL_S3_REGION") or "us-east-1"
            s3 = [g("EXTERNAL_S3_ENDPOINT") or f"https://s3.{region}.amazonaws.com", g("EXTERNAL_S3_BUCKET"), region,
                  g("EXTERNAL_S3_ACCESS_KEY_ID"), g("EXTERNAL_S3_SECRET_ACCESS_KEY"),
                  g("EXTERNAL_S3_FORCE_PATH_STYLE").lower() in ("1", "true", "yes")]
            if s3[1] and self.usable_from_containers("S3", s3[0]) and self.attempt("S3", True, probe_s3, *s3):
                self.exports.update({
                    "LANGFUSE_S3_ENDPOINT": container_url(s3[0]), "LANGFUSE_S3_BUCKET": s3[1], "LANGFUSE_S3_REGION": s3[2],
                    "LANGFUSE_S3_ACCESS_KEY_ID": s3[3], "LANGFUSE_S3_SECRET_ACCESS_KEY": s3[4],
                    "LANGFUSE_S3_FORCE_PATH_STYLE": "true" if s3[5] else "false"})
            else:
                self.selfhost.append("minio")
        else:
            # the gateway (ours or theirs) sends traces to the external Langfuse
            self.exports.update({"LANGFUSE_URL": g("EXTERNAL_LANGFUSE_URL").rstrip("/"),
                                 "LITELLM_LANGFUSE_HOST": container_url(g("EXTERNAL_LANGFUSE_URL").rstrip("/")),
                                 "LANGFUSE_PUBLIC_KEY": g("EXTERNAL_LANGFUSE_PUBLIC_KEY"),
                                 "LANGFUSE_SECRET_KEY": g("EXTERNAL_LANGFUSE_SECRET_KEY")})
        if litellm_ext:
            self.exports.update({"LITELLM_URL": g("EXTERNAL_LITELLM_URL").rstrip("/"),
                                 "LITELLM_MASTER_KEY": g("EXTERNAL_LITELLM_MASTER_KEY")})
        else:
            self.selfhost.append("litellm")
        # a router reachable beyond loopback (e.g. from that proxy) only serves bearers of this key
        published = g("ROUTER_BIND_ADDRESS") not in ("", "127.0.0.1", "localhost")
        if (litellm_ext or published) and not g("ROUTER_API_KEY") and self.persist:
            key = "sk-router-" + secrets.token_hex(24)
            set_env_values({"ROUTER_API_KEY": key}, "Router key for a gateway outside this host (scripts/external_services.py)")
            self.exports["ROUTER_API_KEY"] = key
        self.exports.update({
            "GATEWAY_MODE": "external" if litellm_ext else "selfhosted",
            "LANGFUSE_MODE": "external" if langfuse_ext else "selfhosted",
            "SELFHOST_SERVICES": " ".join(self.selfhost),
            "EXTERNAL_SERVICES": "; ".join(f"{k}: {v}" for k, v in self.external.items()),
        })

    def usable_from_containers(self, service: str, url: str) -> bool:
        if loopback_only(url):
            note(f"  ↻ {service}: {url.split('@')[-1]} listens on this host's loopback only, which containers "
                 "cannot reach: self-hosting it instead")
            return False
        return True


# ------------------------------------------------------------------------------
# External gateway: register this stack's routes on a LiteLLM proxy you run
# ------------------------------------------------------------------------------
def gateway_models(cfg: Dict[str, str]) -> List[Dict]:
    served = cfg.get("SERVED_MODEL_NAME") or "model"
    upstream = cfg["EXTERNAL_LITELLM_UPSTREAM_URL"].rstrip("/")
    params = {"model": f"hosted_vllm/{served}", "api_base": upstream, "api_key": cfg.get("ROUTER_API_KEY") or "none"}
    info = {"llmops_managed": upstream, "mode": "chat",  # one proxy can front several hosts
            "supports_function_calling": cfg.get("MODEL_SUPPORTS_TOOLS") == "true",
            "supports_vision": cfg.get("MODEL_SUPPORTS_VISION") == "true",
            "supports_reasoning": cfg.get("MODEL_SUPPORTS_REASONING") == "true"}
    models = [{"model_name": served, "litellm_params": params, "model_info": info}]
    extra = cfg.get("MODEL_THINKING_EXTRA_BODY") or ""
    if cfg.get("MODEL_SUPPORTS_REASONING") == "true" and extra:
        models.append({"model_name": f"{served}-thinking",
                       "litellm_params": {**params, "extra_body": json.loads(extra)}, "model_info": info})
    return models


def gateway_register(cfg: Dict[str, str]) -> int:
    url, key = cfg["EXTERNAL_LITELLM_URL"].rstrip("/"), cfg["EXTERNAL_LITELLM_MASTER_KEY"]
    models = gateway_models(cfg)
    upstream = cfg["EXTERNAL_LITELLM_UPSTREAM_URL"].rstrip("/")
    removed = remove_models(url, key, managed=upstream)  # replace what an earlier run registered
    for m in models:
        status, body = http("POST", f"{url}/model/new", {"Authorization": f"Bearer {key}"}, m)
        if status != 200:
            note(f"  ✗ registering {m['model_name']} on {url} failed: HTTP {status} {str(body)[:160]}")
            return 1
    # the proof that matters: a request through that proxy reaches the model here
    status, body = http("POST", f"{url}/v1/chat/completions", {"Authorization": f"Bearer {key}"},
                        {"model": models[0]["model_name"], "max_tokens": 8,
                         "messages": [{"role": "user", "content": "Say OK."}]})
    if status != 200:
        note(f"  ✗ {url} cannot reach this host's router at {cfg['EXTERNAL_LITELLM_UPSTREAM_URL']}: "
             f"HTTP {status} {str(body)[:200]}")
        remove_models(url, key, managed=upstream)
        return 1
    note(f"  ✓ Registered {', '.join(m['model_name'] for m in models)} on {url}"
         + (f" (replaced {removed} earlier route(s))" if removed else "") + "; a request through it reached the model")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["resolve", "check", "gateway-register", "gateway-unregister"])
    parser.add_argument("--platform", default=os.getenv("LLMOPS_PLATFORM", "cpu"))
    args = parser.parse_args()
    cfg = config()
    if args.command in ("resolve", "check"):
        resolver = Resolver(cfg, persist=args.command == "resolve")
        try:
            resolver.resolve()
        except Exception as e:  # never fatal: an empty plan runs every bundled service
            note(f"  ↻ Your services could not be checked ({why(e)}): running the bundled ones")
            resolver.exports = {}
        if args.command == "resolve":
            for k, v in resolver.exports.items():
                print(f"export {k}={shlex.quote(v)}")
        return 0
    if not cfg.get("EXTERNAL_LITELLM_URL"):
        note("  ✗ EXTERNAL_LITELLM_URL is not set")
        return 1
    try:
        if args.command == "gateway-register":
            return gateway_register(cfg)
        upstream = (cfg.get("EXTERNAL_LITELLM_UPSTREAM_URL") or "").rstrip("/")
        removed = remove_models(cfg["EXTERNAL_LITELLM_URL"], cfg["EXTERNAL_LITELLM_MASTER_KEY"], managed=upstream) if upstream else 0
        note(f"  ✓ Removed {removed} route(s) this stack had registered on {cfg['EXTERNAL_LITELLM_URL']}")
        return 0
    except Exception as e:
        note(f"  ✗ {cfg['EXTERNAL_LITELLM_URL']}: {why(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
