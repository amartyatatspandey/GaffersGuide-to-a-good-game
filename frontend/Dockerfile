# Frontend: build static bundle and serve with nginx (e.g. for Cloud Run).
# Build from repo root: docker build -f frontend/Dockerfile .
# Optional: set VITE_API_URL at build time for API base URL in production.
FROM node:20-alpine AS builder

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
