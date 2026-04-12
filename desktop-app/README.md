This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Desktop shell (Electron + workspace)

Run the marketing site and Electron window that loads the tactical workspace:

```bash
npm run dev:desktop
```

The Next dev server and `wait-on` use **`NEXT_PORT`** (default **3000**). If port 3000 is already taken:

```bash
NEXT_PORT=3001 npm run dev:desktop
```

Then open `http://127.0.0.1:3001/workspace` in a browser, or rely on Electron (see [`electron/main.cjs`](electron/main.cjs), which reads `NEXT_PORT` / `NEXT_HOST`).

The workspace calls the FastAPI backend at **`NEXT_PUBLIC_BACKEND_URL`** when set, otherwise `http://127.0.0.1:8000`.

From the repository root, `scripts/run_integration_desktop.sh` starts the backend and then `npm run dev:desktop`. It refuses to bind if **8000** is busy (override with **`BACKEND_PORT`**; the script exports **`NEXT_PUBLIC_BACKEND_URL`** when `BACKEND_PORT` is not 8000). If **`NEXT_PORT`** is unset, it picks the first free port in **3000–3010** for Next.js and exports it for Electron.

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result (or the port you chose via `NEXT_PORT`).

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
