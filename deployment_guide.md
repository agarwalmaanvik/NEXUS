# Deployment Guide: NEXUS Live Arena

This guide explains how to deploy the NEXUS Poker Bot to a live server (VPS or Cloud) while maintaining security for your model weights and "alpha" logic.

## 1. Prerequisites
- A Linux VPS (Ubuntu 22.04 recommended) with at least 2GB RAM.
- Python 3.9+ installed.
- Node.js and npm installed (for building the frontend).

## 2. Server Setup (Backend)

1. **Clone the repository** (or upload your files) to the server.
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run with Uvicorn** (Productive mode):
   ```bash
   # Use --host 0.0.0.0 to listen on all interfaces
   uvicorn nexus_server:app --host 0.0.0.0 --port 8000
   ```
   > [!TIP]
   > Use `pm2` or a systemd service to keep the backend running:
   > `pm2 start "uvicorn nexus_server:app --host 0.0.0.0 --port 8000" --name nexus-server`

## 3. Frontend Deployment

1. **Enter the frontend directory**:
   ```bash
   cd web_gui
   ```
2. **Install dependencies**:
   ```bash
   npm install
   ```
3. **Update URL**: Open `src/App.jsx` and change `localhost:8000` to your server's IP address or domain.
4. **Build the production bundle**:
   ```bash
   npm run build
   ```
5. **Serve the static files**: The `dist/` folder can now be served using Nginx or a simple static server like `serve`.

## 4. Security Checklist
- [x] **Weights**: Verified that `checkpoints/` is not accessible via FastAPI.
- [x] **State Filter**: `nexus_server.py` explicitly removes bot cards from the JSON response until the showdown.
- [x] **Alpha**: All decision logic occurs inside `SOTAPokerBot.get_action` on the server. The client only sees the result.

## 5. Reverse Proxy (Recommended)
Use Nginx to handle SSL and proxy requests to the backend:
```nginx
location /ws/ {
    proxy_pass http://localhost:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "Upgrade";
}
```
