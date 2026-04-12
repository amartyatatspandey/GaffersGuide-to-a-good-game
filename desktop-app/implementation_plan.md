# Gaffer's Guide Desktop Workspace Layout

This plan outlines the implementation of the core workspace layout for the Gaffer's Guide desktop software (a telemetry-style tactical analysis dashboard). Since our current project is a Next.js web application, this layout will be built as a dedicated route (`/workspace`) that behaves exactly like a full-viewport Electron application layout.

## Goals
- Build a rigid, `100vh` zero-scroll dashboard layout.
- Implement the requested high-contrast dark mode design system matching the Gaffer's Guide telemetry aesthetic.
- Define layout architecture: Titlebar, Sidebar, and a split Main Dashboard (Dropzone, 2D Map, and Terminal).

## User Review Required

> [!IMPORTANT]
> Since we are working inside your Next.js project, I propose building this application view at a completely new route: `src/app/workspace/page.tsx`. 
> This allows you to navigate to `http://localhost:3000/workspace` to view the desktop application layout without destroying the marketing landing page we just finished on the root route. Does this approach work for you, or would you prefer this to be a standalone component for an actual Electron boilerplate?

## Proposed Changes

### Configuration
#### [MODIFY] `src/app/globals.css`
- Inject the specific layout colors into the Tailwind base layer or CSS variables (`bg-panel: #111a12`, `border-edge: #1a2420`). Alternatively, I will map these using strict Tailwind arbitrary values to ensure pinpoint accuracy (`bg-[#111a12]`).

### Desktop Application UI
#### [NEW] `src/app/workspace/page.tsx`
- The core wrapper. `h-screen w-screen overflow-hidden flex flex-col bg-[#0a0f0a] text-gray-300 font-mono`.

#### [NEW] `src/app/workspace/components/Titlebar.tsx`
- 32px height, drag region enabled (`app-region: drag` for electron compatibility).
- Brand logo on left, current file in center, and mock window controls on the right.

#### [NEW] `src/app/workspace/components/Sidebar.tsx`
- Fixed 240px width.
- Lucide-React powered navigation menus and status toggles.

#### [NEW] `src/app/workspace/components/MainDashboard.tsx`
- The core Flex/Grid engine containing:
  - **Video Section**: Placeholder state with drag & drop dashed borders.
  - **Bottom Split**: 
    - *2D Map Box*: SVGs representing tracking coordinates.
    - *Virtual Coach Terminal*: Integrating the aesthetic of the existing `VirtualCoachTerminal` but optimized for half-width bottom panel metrics.

## Open Questions
- Do you want the newly designed `VirtualCoachTerminal` component imported directly from your marketing page components, or should I create a specialized replica optimized for this new desktop view?

## Verification Plan

### Manual Verification
- Run `npm run dev` and navigate to `/workspace`.
- Validate that the layout occupies exactly 100vh with absolutely zero scrollbars.
- Verify the specific color tokens (#0a0f0a, #111a12, #1a2420, etc.) are applied perfectly across panels.
- Check simulated drag-and-drop region layout responsiveness.
