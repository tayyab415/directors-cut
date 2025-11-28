# YouTube Cookies Setup - Simple Workaround

This is the **easiest workaround** for the HF Spaces YouTube restriction - no external services needed!

## How It Works

By uploading your personal YouTube cookies, yt-dlp can authenticate as a logged-in user, which often bypasses network restrictions.

## Step 1: Get Your Cookies

### Option A: Browser Extension (Easiest)

1. **Chrome/Edge**: Install ["Get cookies.txt LOCALLY"](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
2. **Firefox**: Install ["cookies.txt"](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)

### Option B: Manual Export

1. Open browser DevTools (F12)
2. Go to Application/Storage → Cookies → `https://www.youtube.com`
3. Copy all cookies in Netscape format
4. Save as `youtube_cookies.txt`

## Step 2: Export Cookies

1. **Go to YouTube** (make sure you're logged in)
2. **Click the extension icon**
3. **Select `www.youtube.com`**
4. **Click "Export"** or "Copy"
5. **Save as** `youtube_cookies.txt`

The file should look like:
```
# Netscape HTTP Cookie File
.youtube.com	TRUE	/	TRUE	1234567890	VISITOR_INFO1_LIVE	abc123...
.youtube.com	TRUE	/	TRUE	1234567890	YSC	def456...
```

## Step 3: Upload to HF Space

1. Go to your Space: https://huggingface.co/spaces/tyb343/directors-cut
2. In the **"Create Clip"** tab, expand **"YouTube Cookies (Optional)"**
3. Click **"Upload YouTube Cookies File"**
4. Select your `youtube_cookies.txt` file
5. Wait for confirmation: "✅ Cookies file saved!"

## Step 4: Test

Try downloading a YouTube video - it should work now! 🎉

## Important Notes

- **Cookies expire** after ~2 weeks - you'll need to re-upload them
- **Keep cookies private** - don't share them publicly
- **Works on HF Spaces** - no external services needed
- **Free** - no cost, no setup

## Troubleshooting

- **"Cookies expired"**: Re-export and upload fresh cookies
- **"Still not working"**: Try logging out and back into YouTube, then re-export
- **"File format error"**: Make sure it's in Netscape format (the extension handles this)

## Security

- Cookies are stored in `/tmp/` on the Space (ephemeral)
- They're only used for YouTube downloads
- Don't commit cookies to git (already in .gitignore)

