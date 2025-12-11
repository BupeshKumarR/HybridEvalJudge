# SSL/TLS Certificates

This directory should contain your SSL/TLS certificates for HTTPS.

## Development

For development, you can generate self-signed certificates:

```bash
# Generate self-signed certificate (valid for 365 days)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout key.pem \
  -out cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

## Production

For production, use certificates from a trusted Certificate Authority (CA):

### Option 1: Let's Encrypt (Free)

Use Certbot to obtain free SSL certificates:

```bash
# Install Certbot
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Certificates will be placed in /etc/letsencrypt/live/your-domain.com/
# Copy them to this directory:
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ./cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ./key.pem
```

### Option 2: Commercial CA

Purchase certificates from a commercial CA and place them here:
- `cert.pem` - Your certificate (or fullchain)
- `key.pem` - Your private key

## File Permissions

Ensure proper permissions:

```bash
chmod 644 cert.pem
chmod 600 key.pem
```

## Security Notes

- **Never commit certificates to version control**
- Keep private keys secure
- Rotate certificates before expiration
- Use strong key sizes (minimum 2048-bit RSA or 256-bit ECC)
- Enable OCSP stapling for better performance
- Consider using Certificate Transparency monitoring

## Testing

Test your SSL configuration:

```bash
# Test locally
openssl s_client -connect localhost:443 -servername localhost

# Test online (production)
# Visit: https://www.ssllabs.com/ssltest/
```
