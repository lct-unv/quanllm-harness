# Security policy

## Supported versions

| Version | Supported |
| --- | --- |
| Latest minor release | Yes |
| Older releases | No |

## Reporting a vulnerability

Do not report live API keys, gateway credentials, or student data in a public issue. Use the
repository owner's private security contact or hosting platform security advisory channel.

Supported releases receive fixes on the latest minor line. The Harness treats model text as
untrusted data, validates structured responses and tool arguments, and excludes secrets from run
records. Plugins are disabled by default and are not imported until explicitly enabled. Enabled
in-process plugins execute with the current Python process privileges and therefore must be
reviewed and trusted. A host may require a pre-recorded SHA-256 distribution digest and restrict
declared permissions. Subprocess Tool mode removes shell interpretation, forwards only a small
environment, and bounds time and output, but it does not prevent filesystem or network access.
Run untrusted subprocess plugins in an OS sandbox or container with an unprivileged account and
explicit filesystem/network policy. See `docs/PLUGIN_SECURITY.md` for the threat model.

The gateway credential is read from the ignored `APIKEY` file in the process working directory.
Keep that file private, do not commit it, and restrict its filesystem permissions. The gateway
endpoint is fixed by the package, constructed at runtime, and cannot be supplied by REST clients or
command-line arguments. Its complete plaintext form is excluded from source and release archives.
This avoids accidental static disclosure; it is not a secrecy boundary because a distributed
client must resolve its destination while running.

The bundled server binds to `127.0.0.1` by default. Before exposing it to another host, configure
`QUANLLM_SERVER_TOKEN`, terminate TLS at a trusted reverse proxy, set request/body and connection
limits there, and restrict access to run-record directories. The REST API never accepts gateway
credentials from a request. Browser tokens are kept only in the current DOM and are not written to
local storage or cookies.
