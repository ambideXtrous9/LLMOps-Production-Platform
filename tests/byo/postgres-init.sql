-- byo_rw: owns llmops_ext and may create databases (e.g. "langfuse"); byo_ro: read only
CREATE ROLE byo_rw LOGIN PASSWORD 'byo-rw-pass' CREATEDB;
CREATE ROLE byo_ro LOGIN PASSWORD 'byo-ro-pass';
CREATE DATABASE llmops_ext OWNER byo_rw;
GRANT CONNECT ON DATABASE llmops_ext TO byo_ro;
-- the fixture's own Langfuse and LiteLLM
CREATE DATABASE langfuse_ext;
CREATE DATABASE litellm_ext;
