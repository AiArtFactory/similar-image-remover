# Running the web UI behind a reverse proxy

The built-in server is deliberately small: the Python standard library, plain
HTTP, and HTTP Basic Auth. That is a reasonable fit for a trusted home LAN. It
is not a hardened internet-facing service, and it has no TLS of its own.

A reverse proxy sits in front of this server:

```
phone / laptop ── HTTPS ──> Caddy, Nginx, Tailscale, Cloudflare ── HTTP ──> 127.0.0.1:8799
```

The proxy terminates TLS, owns the public hostname and certificate, and forwards
requests to the Python process. Run the app on loopback so the proxy is the only
reachable surface:

```bash
python3 similar_image_remover.py --web \
    --username me --password 'a long password' \
    --local-only \
    --public-url https://images.example.com/
```

`--public-url` only changes the startup banner; it does not affect routing.

## What the proxy enables

- **HTTPS.** Basic Auth passwords and image bytes are no longer sent in clear
  text. Caddy or Nginx can use a normal Let's Encrypt certificate.
- **Access from outside the house** without opening a router port, using
  Tailscale Serve/Funnel, Cloudflare Tunnel, or a VPN.
- **A stable hostname** and one entry point for several services on the same
  machine.
- **Extra access control** — SSO, IP allowlists, rate limits, request logs.
- **Less exposure.** The Python server stays on `127.0.0.1`, so only the proxy
  can reach it.

If you only use the same Wi-Fi, none of this is necessary. The direct
`http://<lan-ip>:8799/` URL with Basic Auth is simpler and works on the phone.

## The native folder dialog and proxies

The **Open Finder… / Open Explorer…** button runs a dialog on the *server's*
desktop. It must never be offered to a remote phone. Detection works like this:

- Loopback is trusted as a proxy by default. Forwarded headers from loopback are
  parsed to find the real client; other proxy addresses can be trusted with
  `--trusted-proxy`.
- A request from a mobile user agent is treated as remote even when a NAT makes
  it look like loopback (the Android emulator's `10.0.2.2` does exactly this).
- A request carrying proxy headers but no trusted client address is remote for
  the native picker.
- `--no-native-picker` disables the native dialog entirely if your proxy strips
  headers or you simply do not want OS dialogs.

The phone uses the in-page folder browser instead. It lists **the server's**
directories and never touches the phone's storage.

If you are physically at the server machine, open
`http://127.0.0.1:8799/` directly to get the native dialog. The proxy URL will
offer the in-page browser.

## Caddy

The simplest HTTPS setup. Caddy obtains and renews the certificate.

```caddyfile
images.example.com {
    reverse_proxy 127.0.0.1:8799
}
```

Start the app with `--local-only --public-url https://images.example.com/`.
Caddy preserves the browser's `Host` and sets `X-Forwarded-For`,
`X-Forwarded-Proto`, and `X-Forwarded-Host`.

## Nginx

At a hostname root:

```nginx
server {
    listen 443 ssl;
    server_name images.example.com;

    ssl_certificate     /etc/letsencrypt/live/images.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/images.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8799;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
    }
}
```

Under a **subpath**, keep the trailing slash and strip the prefix:

```nginx
location /similar/ {
    proxy_pass http://127.0.0.1:8799/;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Host $host;
}
```

The UI uses relative API and image URLs specifically so `/similar/` works when
the proxy strips the prefix. Visit `https://host/similar/` **with the trailing
slash**; a path without it has no directory to resolve relative URLs against.

For proxies on a non-loopback address, tell the app to trust their forwarded
headers:

```bash
--trusted-proxy 10.0.0.5/32
```

Do not add a proxy you do not control.

## Tailscale Serve

Tailscale Serve is a private reverse proxy for your tailnet. The exact command
varies by Tailscale version; `tailscale serve --help` is authoritative. A common
form is:

```bash
tailscale serve --bg http://127.0.0.1:8799
```

It forwards to the local server and exposes an HTTPS name such as
`https://your-machine.your-tailnet.ts.net/`. Because the request arrives through a proxy,
the phone gets the in-page folder browser and **not** Finder. Use
`http://127.0.0.1:8799/` on the Mac itself when you want the native dialog.
`tailscale funnel` makes it public; think carefully before using it for a tool
that permanently deletes files.

## Cloudflare Tunnel

For a public HTTPS hostname without port forwarding:

```bash
cloudflared tunnel --url http://127.0.0.1:8799
```

Add Cloudflare Access or another identity policy at the edge. The app's Basic
Auth remains a second factor, but do not rely on it alone for a public endpoint.

## Security checklist

- Use HTTPS for anything beyond a trusted LAN.
- Use a long, unique password; `--password` is visible in shell history, so
  prefer `SIR_PASSWORD`.
- Keep the app on `--local-only` and expose only the proxy.
- Do not expose the raw Python port to the internet.
- The in-page browser lists the server's directory names to any authenticated
  user. With `--no-auth`, that includes anyone who can reach the proxy.
- `--no-auth` is for a short, trusted session only.
- The tool deletes permanently (`os.remove`). Pins and the server-side
  “never empty a group” check reduce mistakes; they are not backups.
