# YouTube Proxy Service Setup

This proxy service allows you to bypass Hugging Face Spaces' network restrictions on YouTube downloads.

## Quick Deploy Options

### Option 1: Railway (Recommended - Free tier available)

1. Go to [Railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Add a new service → "Empty Service"
5. In the service settings:
   - Set the start command: `python youtube_proxy_service.py`
   - Add environment variable: `PORT=5000`
6. Railway will auto-detect Python and install dependencies
7. Copy the public URL (e.g., `https://your-app.railway.app`)

### Option 2: Render (Free tier available)

1. Go to [Render.com](https://render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `youtube-proxy`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python youtube_proxy_service.py`
   - **Port**: `5000`
5. Add environment variable: `PORT=5000`
6. Deploy and copy the service URL

### Option 3: Fly.io (Free tier available)

1. Install Fly CLI: `curl -L https://fly.io/install.sh | sh`
2. Run: `fly launch`
3. Create `fly.toml`:
```toml
app = "youtube-proxy"
[build]
  builder = "paketobuildpacks/builder:base"
[env]
  PORT = "8080"
[[services]]
  internal_port = 8080
  protocol = "tcp"
```
4. Deploy: `fly deploy`

### Option 4: Local Development

```bash
pip install flask flask-cors yt-dlp requests
python youtube_proxy_service.py
```

The service will run on `http://localhost:5000`

## Configure Your HF Space

Once the proxy service is deployed:

1. Go to your HF Space settings: https://huggingface.co/spaces/tyb343/directors-cut/settings
2. Go to "Variables and secrets"
3. Add a new variable:
   - **Name**: `YOUTUBE_PROXY_URL`
   - **Value**: Your proxy service URL (e.g., `https://your-app.railway.app`)
4. Save and rebuild the Space

## Testing

Test the proxy service:

```bash
# Health check
curl https://your-proxy-url.com/health

# Get video info
curl -X POST https://your-proxy-url.com/api/video-info \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

## Security Notes

- The proxy service is public by default
- Consider adding authentication if you want to restrict access
- For production, add rate limiting and request validation
- Monitor usage to avoid abuse

## Troubleshooting

- **CORS errors**: Make sure `ALLOWED_ORIGINS` includes your HF Space domain
- **Timeouts**: Increase timeout values for long videos
- **Memory issues**: The service downloads files to temp, ensure sufficient disk space

