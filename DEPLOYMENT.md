
# Production Deployment Guide

## Prerequisites
1. PostgreSQL database configured (automatic on Replit)
2. Exchange API credentials in Replit Secrets (for real trading)
3. Replit Core subscription (recommended for production)

## Environment Variables

### Required (Auto-configured on Replit)
- `DATABASE_URL` - PostgreSQL connection string
- `PGHOST`, `PGUSER`, `PGPASSWORD`, `PGDATABASE` - Database credentials

### Optional (Add in Secrets for real trading)
- `BINANCE_API_KEY` - Binance API key
- `BINANCE_API_SECRET` - Binance API secret
- `BYBIT_API_KEY` - Bybit API key
- `BYBIT_API_SECRET` - Bybit API secret

### Trading Configuration
- `EXCHANGE` - Exchange to use (binance/bybit) - default: binance
- `TRADING_MODE` - Mode (virtual/real) - default: virtual

## Deployment Steps

1. **Configure Secrets**
   - Click 🔒 Secrets in left sidebar
   - Add your API credentials (if using real mode)
   - Save and restart

2. **Deploy**
   - Click "Deploy" button
   - Choose "Autoscale Deployments" for production
   - Configure custom domain (optional)
   - Deploy

3. **Monitor**
   - Check logs in Deployments tab
   - Monitor performance metrics
   - Set up alerts for errors

## Production Checklist

- [ ] Database properly configured
- [ ] API credentials added to Secrets (if real trading)
- [ ] Deployment type selected (Autoscale recommended)
- [ ] Custom domain configured (optional)
- [ ] Error logging verified
- [ ] Rate limits configured
- [ ] Health checks working

## Safety Features

- Virtual mode for testing (no real trades)
- Position limits and risk management
- Emergency stop functionality
- Daily loss limits
- Rate limiting on API calls

## Monitoring

- Check application logs for errors
- Monitor database performance
- Track API rate limits
- Review trading performance metrics

## Scaling

For high traffic, Autoscale Deployments will:
- Scale horizontally based on demand
- Handle multiple concurrent users
- Provide 99.95% uptime SLA
