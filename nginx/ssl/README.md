# SSL Certificates

Place your SSL certificate and private key here before running Docker:

    nginx/ssl/fullchain.pem   — your certificate (+ chain if using Let's Encrypt)
    nginx/ssl/privkey.pem     — your private key

## Local development (self-signed)

Run this once from the AI/ project root:

    mkdir -p nginx/ssl
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/privkey.pem \
        -out nginx/ssl/fullchain.pem \
        -subj "/CN=localhost"

Then start with: docker compose up --build
Your browser will warn about the self-signed cert — ignore it for local testing.

## Production (Let's Encrypt / Certbot)

    certbot certonly --standalone -d your-domain.com
    cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/
    cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/

Renew yearly (or automate with a cron job):
    certbot renew
    docker compose restart nginx
