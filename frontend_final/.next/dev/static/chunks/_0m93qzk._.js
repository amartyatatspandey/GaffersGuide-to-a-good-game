(globalThis["TURBOPACK"] || (globalThis["TURBOPACK"] = [])).push([typeof document === "object" ? document.currentScript : undefined,
"[project]/app/workspace/components/Titlebar.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "Titlebar",
    ()=>Titlebar
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$minus$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Minus$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/minus.mjs [app-client] (ecmascript) <export default as Minus>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$square$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Square$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/square.mjs [app-client] (ecmascript) <export default as Square>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$x$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__X$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/x.mjs [app-client] (ecmascript) <export default as X>");
;
;
function Titlebar() {
    const handleMinimize = ()=>{
        if (window.gaffersGuide?.windowControls?.minimize) {
            window.gaffersGuide.windowControls.minimize();
        }
    };
    const handleMaximize = ()=>{
        if (window.gaffersGuide?.windowControls?.maximize) {
            window.gaffersGuide.windowControls.maximize();
        } else {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen().catch((err)=>{
                    console.log(`Error attempting to enable fullscreen: ${err.message}`);
                });
            } else {
                document.exitFullscreen();
            }
        }
    };
    const handleClose = ()=>{
        if (window.gaffersGuide?.windowControls?.close) {
            window.gaffersGuide.windowControls.close();
        } else {
            window.close();
        }
    };
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "h-8 bg-black flex items-center justify-between border-b border-[#1a2420] select-none shrink-0",
        style: {
            WebkitAppRegion: 'drag'
        },
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "pl-4 flex items-center h-full",
                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                    className: "text-xs font-bold font-mono tracking-tight text-gray-300",
                    children: [
                        "Gaffers",
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                            className: "text-neon",
                            children: "."
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/components/Titlebar.tsx",
                            lineNumber: 37,
                            columnNumber: 18
                        }, this),
                        "Guide"
                    ]
                }, void 0, true, {
                    fileName: "[project]/app/workspace/components/Titlebar.tsx",
                    lineNumber: 36,
                    columnNumber: 9
                }, this)
            }, void 0, false, {
                fileName: "[project]/app/workspace/components/Titlebar.tsx",
                lineNumber: 35,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "flex-1 flex justify-center items-center h-full",
                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                    className: "text-xs text-gray-500 font-mono tracking-wider",
                    children: "Match_Analysis_Final.mp4"
                }, void 0, false, {
                    fileName: "[project]/app/workspace/components/Titlebar.tsx",
                    lineNumber: 41,
                    columnNumber: 9
                }, this)
            }, void 0, false, {
                fileName: "[project]/app/workspace/components/Titlebar.tsx",
                lineNumber: 40,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "flex h-full",
                style: {
                    WebkitAppRegion: 'no-drag'
                },
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                        onClick: handleMinimize,
                        className: "h-full px-4 hover:bg-gray-800 text-gray-400 transition-colors flex items-center justify-center",
                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$minus$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Minus$3e$__["Minus"], {
                            size: 14
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/components/Titlebar.tsx",
                            lineNumber: 45,
                            columnNumber: 11
                        }, this)
                    }, void 0, false, {
                        fileName: "[project]/app/workspace/components/Titlebar.tsx",
                        lineNumber: 44,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                        onClick: handleMaximize,
                        className: "h-full px-4 hover:bg-gray-800 text-gray-400 transition-colors flex items-center justify-center",
                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$square$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Square$3e$__["Square"], {
                            size: 12
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/components/Titlebar.tsx",
                            lineNumber: 48,
                            columnNumber: 11
                        }, this)
                    }, void 0, false, {
                        fileName: "[project]/app/workspace/components/Titlebar.tsx",
                        lineNumber: 47,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                        onClick: handleClose,
                        className: "h-full px-4 hover:bg-red-600 hover:text-white text-gray-400 transition-colors flex items-center justify-center",
                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$x$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__X$3e$__["X"], {
                            size: 16
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/components/Titlebar.tsx",
                            lineNumber: 51,
                            columnNumber: 11
                        }, this)
                    }, void 0, false, {
                        fileName: "[project]/app/workspace/components/Titlebar.tsx",
                        lineNumber: 50,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/Titlebar.tsx",
                lineNumber: 43,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/app/workspace/components/Titlebar.tsx",
        lineNumber: 34,
        columnNumber: 5
    }, this);
}
_c = Titlebar;
var _c;
__turbopack_context__.k.register(_c, "Titlebar");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/app/workspace/components/EngineSettingsModal.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "EngineSettingsModal",
    ()=>EngineSettingsModal
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$x$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__X$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/x.mjs [app-client] (ecmascript) <export default as X>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$cloud$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Cloud$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/cloud.mjs [app-client] (ecmascript) <export default as Cloud>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$lock$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Lock$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/lock.mjs [app-client] (ecmascript) <export default as Lock>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$cpu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Cpu$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/cpu.mjs [app-client] (ecmascript) <export default as Cpu>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$server$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Server$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/server.mjs [app-client] (ecmascript) <export default as Server>");
;
var _s = __turbopack_context__.k.signature();
"use client";
;
;
const LS_ENGINE = "gaffer-engine-type";
const LS_CONNECTIVITY = "gaffer-connectivity";
const LS_OLLAMA_MODEL = "gaffer-ollama-model";
const LS_CLOUD_API_KEY = "gaffer-cloud-api-key";
function EngineSettingsModal({ isOpen, onClose }) {
    _s();
    const [engineType, setEngineType] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])('local');
    const [connectivity, setConnectivity] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])('online');
    const [ollamaModel, setOllamaModel] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])("llama3");
    const [cloudApiKey, setCloudApiKey] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])("");
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "EngineSettingsModal.useEffect": ()=>{
            if (!isOpen || ("TURBOPACK compile-time value", "object") === "undefined") return;
            const eng = localStorage.getItem(LS_ENGINE);
            if (eng === "local" || eng === "cloud") setEngineType(eng);
            const con = localStorage.getItem(LS_CONNECTIVITY);
            if (con === "online" || con === "offline") setConnectivity(con);
            const m = localStorage.getItem(LS_OLLAMA_MODEL);
            if (m && m.trim()) setOllamaModel(m.trim());
            const key = localStorage.getItem(LS_CLOUD_API_KEY);
            if (key) setCloudApiKey(key);
        }
    }["EngineSettingsModal.useEffect"], [
        isOpen
    ]);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "EngineSettingsModal.useEffect": ()=>{
            if ("TURBOPACK compile-time falsy", 0) //TURBOPACK unreachable
            ;
            localStorage.setItem(LS_ENGINE, engineType);
        }
    }["EngineSettingsModal.useEffect"], [
        engineType
    ]);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "EngineSettingsModal.useEffect": ()=>{
            if ("TURBOPACK compile-time falsy", 0) //TURBOPACK unreachable
            ;
            localStorage.setItem(LS_CONNECTIVITY, connectivity);
        }
    }["EngineSettingsModal.useEffect"], [
        connectivity
    ]);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "EngineSettingsModal.useEffect": ()=>{
            if ("TURBOPACK compile-time falsy", 0) //TURBOPACK unreachable
            ;
            localStorage.setItem(LS_OLLAMA_MODEL, ollamaModel.trim() || "llama3");
        }
    }["EngineSettingsModal.useEffect"], [
        ollamaModel
    ]);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "EngineSettingsModal.useEffect": ()=>{
            if ("TURBOPACK compile-time falsy", 0) //TURBOPACK unreachable
            ;
            localStorage.setItem(LS_CLOUD_API_KEY, cloudApiKey.trim());
        }
    }["EngineSettingsModal.useEffect"], [
        cloudApiKey
    ]);
    if (!isOpen) return null;
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm font-sans transition-all duration-300",
        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
            className: "bg-[#111a12] border border-gray-800 rounded-2xl w-full max-w-xl shadow-[0_0_50px_rgba(0,0,0,0.8)] overflow-hidden relative",
            children: [
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: "flex justify-between items-center bg-[#0a0f0a] px-6 py-4 border-b border-gray-900",
                    children: [
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                            className: "text-sm font-bold text-gray-300 font-mono tracking-widest uppercase flex items-center gap-2",
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$cpu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Cpu$3e$__["Cpu"], {
                                    size: 16,
                                    className: "text-emerald-500"
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                    lineNumber: 56,
                                    columnNumber: 13
                                }, this),
                                " AI Engine Configuration"
                            ]
                        }, void 0, true, {
                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                            lineNumber: 55,
                            columnNumber: 11
                        }, this),
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                            onClick: onClose,
                            className: "text-gray-500 hover:text-white transition-colors",
                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$x$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__X$3e$__["X"], {
                                size: 18
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                lineNumber: 59,
                                columnNumber: 13
                            }, this)
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                            lineNumber: 58,
                            columnNumber: 11
                        }, this)
                    ]
                }, void 0, true, {
                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                    lineNumber: 54,
                    columnNumber: 9
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: "p-6",
                    children: [
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                            className: "text-sm text-gray-400 mb-6",
                            children: "Configure your primary inference engine and connectivity state to dictate how telemetry is processed."
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                            lineNumber: 65,
                            columnNumber: 11
                        }, this),
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "mb-4 text-[10px] font-bold text-gray-500 uppercase tracking-widest font-mono",
                            children: "1. Inference Architecture"
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                            lineNumber: 68,
                            columnNumber: 11
                        }, this),
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "flex gap-2 p-1 bg-[#0a0f0a] border border-gray-800 rounded-xl mb-6",
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                    onClick: ()=>setEngineType('cloud'),
                                    className: `flex-1 py-3 px-4 rounded-lg flex items-center justify-center gap-2 text-sm font-bold transition-all border ${engineType === 'cloud' ? 'bg-blue-900/40 text-blue-400 border-blue-800 shadow-[0_0_15px_rgba(59,130,246,0.2)]' : 'text-gray-500 border-transparent hover:text-gray-300 hover:bg-gray-800/50'}`,
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$cloud$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Cloud$3e$__["Cloud"], {
                                            size: 16
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                            lineNumber: 78,
                                            columnNumber: 15
                                        }, this),
                                        " Cloud Engine"
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                    lineNumber: 70,
                                    columnNumber: 13
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                    onClick: ()=>setEngineType('local'),
                                    className: `flex-1 py-3 px-4 rounded-lg flex items-center justify-center gap-2 text-sm font-bold transition-all border ${engineType === 'local' ? 'bg-emerald-900/40 text-emerald-400 border-emerald-800 shadow-[0_0_15px_rgba(16,185,129,0.2)]' : 'text-gray-500 border-transparent hover:text-gray-300 hover:bg-gray-800/50'}`,
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$lock$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Lock$3e$__["Lock"], {
                                            size: 16
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                            lineNumber: 88,
                                            columnNumber: 15
                                        }, this),
                                        " Local Engine"
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                    lineNumber: 80,
                                    columnNumber: 13
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                            lineNumber: 69,
                            columnNumber: 11
                        }, this),
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "mb-4 text-[10px] font-bold text-gray-500 uppercase tracking-widest font-mono",
                            children: "2. Connectivity Mode"
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                            lineNumber: 93,
                            columnNumber: 11
                        }, this),
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "flex gap-2 p-1 bg-[#0a0f0a] border border-gray-800 rounded-xl mb-6",
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                    onClick: ()=>setConnectivity('online'),
                                    className: `flex-1 py-2 px-4 rounded-lg flex items-center justify-center gap-2 text-xs font-bold transition-all border ${connectivity === 'online' ? 'bg-gray-800 text-gray-200 border-gray-600' : 'text-gray-600 border-transparent hover:text-gray-400 hover:bg-gray-900'}`,
                                    children: "Online Mode"
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                    lineNumber: 95,
                                    columnNumber: 13
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                    onClick: ()=>setConnectivity('offline'),
                                    className: `flex-1 py-2 px-4 rounded-lg flex items-center justify-center gap-2 text-xs font-bold transition-all border ${connectivity === 'offline' ? 'bg-gray-800 text-gray-200 border-gray-600' : 'text-gray-600 border-transparent hover:text-gray-400 hover:bg-gray-900'}`,
                                    children: "Offline Mode"
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                    lineNumber: 105,
                                    columnNumber: 13
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                            lineNumber: 94,
                            columnNumber: 11
                        }, this),
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "min-h-[140px] p-5 bg-[#0a0f0a]/50 rounded-xl border border-gray-800/50",
                            children: [
                                engineType === 'cloud' && connectivity === 'online' && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "animate-fade-in-up",
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                            className: "block text-xs font-bold text-gray-500 uppercase tracking-widest mb-2 font-mono",
                                            children: "OpenAI / Gemini API Key"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                            lineNumber: 121,
                                            columnNumber: 17
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "relative",
                                            children: [
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                                    type: "password",
                                                    value: cloudApiKey,
                                                    onChange: (e)=>setCloudApiKey(e.target.value),
                                                    placeholder: "sk-...",
                                                    className: "w-full bg-[#0a0f0a] border border-gray-700 rounded-lg px-4 py-3 text-gray-200 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/50 transition-all font-mono text-sm"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                                    lineNumber: 123,
                                                    columnNumber: 19
                                                }, this),
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$server$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Server$3e$__["Server"], {
                                                    size: 16,
                                                    className: "absolute right-4 top-1/2 -translate-y-1/2 text-gray-600"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                                    lineNumber: 130,
                                                    columnNumber: 19
                                                }, this)
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                            lineNumber: 122,
                                            columnNumber: 17
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                            className: "text-xs text-blue-500/70 mt-3 flex items-center gap-1 font-mono",
                                            children: [
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$cloud$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Cloud$3e$__["Cloud"], {
                                                    size: 12
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                                    lineNumber: 132,
                                                    columnNumber: 96
                                                }, this),
                                                " Live streaming telemetry to cloud endpoints."
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                            lineNumber: 132,
                                            columnNumber: 17
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                    lineNumber: 120,
                                    columnNumber: 15
                                }, this),
                                engineType === 'cloud' && connectivity === 'offline' && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "animate-fade-in-up py-4 text-center",
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$cloud$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Cloud$3e$__["Cloud"], {
                                            size: 32,
                                            className: "mx-auto text-indigo-500/50 mb-3"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                            lineNumber: 138,
                                            columnNumber: 18
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h4", {
                                            className: "text-sm font-bold text-indigo-400 mb-1",
                                            children: "Cloud Mock / Cached Sync"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                            lineNumber: 139,
                                            columnNumber: 18
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                            className: "text-xs text-gray-500 max-w-sm mx-auto",
                                            children: "Operating without internet. Telemetry is parsing against the last synced high-fidelity cached payloads."
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                            lineNumber: 140,
                                            columnNumber: 18
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                    lineNumber: 137,
                                    columnNumber: 15
                                }, this),
                                engineType === 'local' && connectivity === 'online' && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "animate-fade-in-up py-4 space-y-3 text-center",
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$server$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Server$3e$__["Server"], {
                                            size: 32,
                                            className: "mx-auto text-cyan-500/50 mb-3"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                            lineNumber: 146,
                                            columnNumber: 18
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h4", {
                                            className: "text-sm font-bold text-cyan-400 mb-1",
                                            children: "Distributed Local Node"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                            lineNumber: 147,
                                            columnNumber: 18
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                            className: "text-xs text-gray-500 max-w-sm mx-auto",
                                            children: [
                                                "Chat uses Ollama at ",
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                    className: "font-mono text-gray-400",
                                                    children: "OLLAMA_BASE_URL"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                                    lineNumber: 148,
                                                    columnNumber: 92
                                                }, this),
                                                " (default ",
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                    className: "font-mono",
                                                    children: "http://127.0.0.1:11434"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                                    lineNumber: 148,
                                                    columnNumber: 166
                                                }, this),
                                                ")."
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                            lineNumber: 148,
                                            columnNumber: 18
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "text-left pt-2",
                                            children: [
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                                    className: "block text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-1 font-mono",
                                                    children: "Ollama model name"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                                    lineNumber: 150,
                                                    columnNumber: 20
                                                }, this),
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                                    type: "text",
                                                    value: ollamaModel,
                                                    onChange: (e)=>setOllamaModel(e.target.value),
                                                    placeholder: "llama3",
                                                    className: "w-full bg-[#0a0f0a] border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 font-mono focus:outline-none focus:border-cyan-600"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                                    lineNumber: 151,
                                                    columnNumber: 20
                                                }, this)
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                            lineNumber: 149,
                                            columnNumber: 18
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                    lineNumber: 145,
                                    columnNumber: 15
                                }, this),
                                engineType === 'local' && connectivity === 'offline' && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "animate-fade-in-up",
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                            className: "block text-xs font-bold text-gray-500 uppercase tracking-widest mb-2 font-mono",
                                            children: "Strict Airgap Checkpoint"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                            lineNumber: 164,
                                            columnNumber: 17
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("select", {
                                            value: ollamaModel,
                                            onChange: (e)=>setOllamaModel(e.target.value),
                                            className: "w-full bg-[#0a0f0a] border border-gray-700 rounded-lg px-4 py-3 text-gray-200 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500/50 transition-all font-mono text-sm appearance-none custom-select",
                                            children: [
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("option", {
                                                    value: "llama3",
                                                    children: "llama3 (Recommended)"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                                    lineNumber: 170,
                                                    columnNumber: 19
                                                }, this),
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("option", {
                                                    value: "llama3:8b",
                                                    children: "llama3:8b"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                                    lineNumber: 171,
                                                    columnNumber: 19
                                                }, this),
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("option", {
                                                    value: "mistral",
                                                    children: "mistral"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                                    lineNumber: 172,
                                                    columnNumber: 19
                                                }, this),
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("option", {
                                                    value: "phi3",
                                                    children: "phi3"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                                    lineNumber: 173,
                                                    columnNumber: 19
                                                }, this)
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                            lineNumber: 165,
                                            columnNumber: 17
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "mt-4 p-3 bg-amber-900/20 border border-amber-800/50 rounded-lg flex items-start gap-3",
                                            children: [
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                    className: "text-amber-500 mt-0.5",
                                                    children: "⚠️"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                                    lineNumber: 177,
                                                    columnNumber: 19
                                                }, this),
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                                    className: "text-xs text-amber-200/80 leading-relaxed font-mono",
                                                    children: [
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("strong", {
                                                            children: "Airgap Active:"
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                                            lineNumber: 179,
                                                            columnNumber: 21
                                                        }, this),
                                                        " Running entirely on local device hardware. Ensure you have 16GB+ RAM available to prevent thermal throttling."
                                                    ]
                                                }, void 0, true, {
                                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                                    lineNumber: 178,
                                                    columnNumber: 19
                                                }, this)
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                            lineNumber: 176,
                                            columnNumber: 17
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                    lineNumber: 163,
                                    columnNumber: 15
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                            lineNumber: 118,
                            columnNumber: 11
                        }, this)
                    ]
                }, void 0, true, {
                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                    lineNumber: 64,
                    columnNumber: 9
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: "bg-[#0a0f0a] border-t border-gray-900 px-6 py-4 flex justify-between items-center",
                    children: [
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "font-mono text-[10px] uppercase text-gray-600 tracking-widest",
                            children: [
                                "Target: ",
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                    className: engineType === 'cloud' && connectivity === 'online' ? 'text-blue-500' : engineType === 'cloud' && connectivity === 'offline' ? 'text-indigo-400' : engineType === 'local' && connectivity === 'online' ? 'text-cyan-400' : 'text-emerald-500',
                                    children: [
                                        engineType,
                                        " • ",
                                        connectivity
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                                    lineNumber: 191,
                                    columnNumber: 21
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                            lineNumber: 190,
                            columnNumber: 11
                        }, this),
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                            type: "button",
                            onClick: ()=>{
                                if ("TURBOPACK compile-time truthy", 1) {
                                    window.dispatchEvent(new Event("gaffer-engine-changed"));
                                }
                                onClose();
                            },
                            className: `rounded px-6 py-2 text-sm font-bold tracking-wide transition-colors ${engineType === 'cloud' ? 'bg-blue-600 hover:bg-blue-500 text-white' : 'bg-emerald-600 hover:bg-emerald-500 text-white'}`,
                            children: "Deploy Pipeline"
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                            lineNumber: 197,
                            columnNumber: 11
                        }, this)
                    ]
                }, void 0, true, {
                    fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
                    lineNumber: 189,
                    columnNumber: 9
                }, this)
            ]
        }, void 0, true, {
            fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
            lineNumber: 52,
            columnNumber: 7
        }, this)
    }, void 0, false, {
        fileName: "[project]/app/workspace/components/EngineSettingsModal.tsx",
        lineNumber: 51,
        columnNumber: 5
    }, this);
}
_s(EngineSettingsModal, "75K5t/iGB5ynGUotaWdv7erVL8I=");
_c = EngineSettingsModal;
var _c;
__turbopack_context__.k.register(_c, "EngineSettingsModal");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/lib/apiBase.ts [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "getApiBaseUrl",
    ()=>getApiBaseUrl,
    "getAuthHeaders",
    ()=>getAuthHeaders,
    "getWsBaseUrl",
    ()=>getWsBaseUrl
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$build$2f$polyfills$2f$process$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = /*#__PURE__*/ __turbopack_context__.i("[project]/node_modules/next/dist/build/polyfills/process.js [app-client] (ecmascript)");
function getApiBaseUrl() {
    if ("TURBOPACK compile-time truthy", 1) {
        const bridge = window.gaffersGuide?.getApiBase;
        if (typeof bridge === 'function') {
            const fromElectron = bridge();
            if (typeof fromElectron === 'string' && fromElectron.trim().length > 0) {
                return fromElectron.trim().replace(/\/$/, '');
            }
        }
    }
    const env = ("TURBOPACK compile-time value", "http://127.0.0.1:8000");
    if (typeof env === 'string' && env.trim().length > 0) {
        return env.trim().replace(/\/$/, '');
    }
    return 'http://127.0.0.1:8000';
}
function getWsBaseUrl() {
    const httpUrl = getApiBaseUrl();
    if (httpUrl.startsWith('https://')) {
        return httpUrl.replace('https://', 'wss://');
    }
    return httpUrl.replace('http://', 'ws://');
}
function getAuthHeaders() {
    const headers = {};
    const apiKey = ("TURBOPACK compile-time value", "gG_83a9f4c7b2d5e11496a80275819d4b3f");
    if ("TURBOPACK compile-time truthy", 1) {
        headers["X-API-Key"] = apiKey;
    }
    if ("TURBOPACK compile-time truthy", 1) {
        const cloudApiKey = localStorage.getItem("gaffer-cloud-api-key");
        if (cloudApiKey) {
            headers["X-LLM-API-Key"] = cloudApiKey;
        }
    }
    return headers;
}
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/lib/debugSessionLog.ts [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

/**
 * NDJSON append via same-origin `/api/debug-log` (writes in dev; route no-ops in production).
 */ __turbopack_context__.s([
    "debugSessionLog",
    ()=>debugSessionLog
]);
function debugSessionLog(entry) {
    const payload = JSON.stringify({
        ...entry,
        timestamp: Date.now()
    });
    fetch("/api/debug-log", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: payload
    }).catch(()=>{});
}
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/lib/api/jobs.ts [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "createJob",
    ()=>createJob,
    "deleteReport",
    ()=>deleteReport,
    "getJobEvents",
    ()=>getJobEvents,
    "getReport",
    ()=>getReport,
    "getTracking",
    ()=>getTracking,
    "listReports",
    ()=>listReports,
    "saveReport",
    ()=>saveReport
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$build$2f$polyfills$2f$process$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = /*#__PURE__*/ __turbopack_context__.i("[project]/node_modules/next/dist/build/polyfills/process.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/apiBase.ts [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$debugSessionLog$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/debugSessionLog.ts [app-client] (ecmascript)");
;
;
/** Local Modal-less dev: default `local` unless `NEXT_PUBLIC_CV_ENGINE=cloud`. */ const defaultCvEngine = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$build$2f$polyfills$2f$process$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"].env.NEXT_PUBLIC_CV_ENGINE === 'cloud' ? 'cloud' : 'local';
async function createJob(file, llmEngine = 'local', qualityProfile = 'balanced', chunkingInterval = '15-minute intervals') {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('cv_engine', defaultCvEngine);
    formData.append('llm_engine', llmEngine);
    formData.append('quality_profile', qualityProfile);
    formData.append('chunking_interval', chunkingInterval);
    const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
    const res = await fetch(`${base}/api/v1/jobs`, {
        method: 'POST',
        headers: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])(),
        body: formData
    });
    // #region agent log
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$debugSessionLog$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["debugSessionLog"])({
        sessionId: 'bb63ae',
        hypothesisId: 'H3-H4',
        location: 'jobs.ts:createJob',
        message: 'createJob response',
        data: {
            apiBaseHost: (()=>{
                try {
                    return new URL(base).host;
                } catch  {
                    return 'invalid-url';
                }
            })(),
            ok: res.ok,
            status: res.status,
            cv_engine: defaultCvEngine
        }
    });
    // #endregion
    if (!res.ok) {
        throw new Error(`Failed to create job: ${res.statusText}`);
    }
    return res.json();
}
const _sleep = (ms)=>new Promise((r)=>setTimeout(r, ms));
async function getTracking(jobId) {
    const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
    const url = `${base}/api/v1/jobs/${jobId}/tracking`;
    const maxAttempts = 48;
    for(let attempt = 0; attempt < maxAttempts; attempt++){
        const res = await fetch(url, {
            headers: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
        });
        // #region agent log
        (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$debugSessionLog$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["debugSessionLog"])({
            sessionId: 'bb63ae',
            runId: 'post-fix',
            hypothesisId: 'H1-H3',
            location: 'jobs.ts:getTracking',
            message: 'getTracking response',
            data: {
                jobIdPrefix: jobId.slice(0, 8),
                attempt,
                ok: res.ok,
                status: res.status
            }
        });
        // #endregion
        if (res.ok) {
            return await res.json();
        }
        if (res.status === 425 && attempt < maxAttempts - 1) {
            await _sleep(Math.min(2000, 350 + attempt * 120));
            continue;
        }
        throw new Error(res.status === 425 ? 'Tracking file was not ready after waiting for the pipeline (HTTP 425).' : `Tracking not ready or not found (HTTP ${res.status})`);
    }
    throw new Error('Tracking not ready or not found');
}
async function listReports() {
    const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
    const res = await fetch(`${base}/api/v1/elite/reports`, {
        headers: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
    });
    if (!res.ok) throw new Error('Failed to list reports');
    return res.json();
}
async function getReport(reportId) {
    const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
    const res = await fetch(`${base}/api/v1/elite/reports/${reportId}`, {
        headers: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
    });
    if (!res.ok) throw new Error('Failed to fetch report');
    return res.json();
}
async function saveReport(reportData) {
    const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
    const res = await fetch(`${base}/api/v1/elite/reports/save`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            ...(0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
        },
        body: JSON.stringify(reportData)
    });
    if (!res.ok) throw new Error('Failed to save report');
    return res.json();
}
async function deleteReport(reportId) {
    const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
    const res = await fetch(`${base}/api/v1/elite/reports/${reportId}`, {
        method: 'DELETE',
        headers: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
    });
    if (!res.ok) throw new Error('Failed to delete report');
    return res.json();
}
async function getJobEvents(jobId) {
    const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
    const res = await fetch(`${base}/api/v1/jobs/${jobId}/events`, {
        headers: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
    });
    if (!res.ok) throw new Error('Failed to fetch job events');
    return res.json();
}
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/app/workspace/components/ProfileModal.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "ProfileModal",
    ()=>ProfileModal
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$x$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__X$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/x.mjs [app-client] (ecmascript) <export default as X>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$user$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__User$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/user.mjs [app-client] (ecmascript) <export default as User>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$shield$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Shield$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/shield.mjs [app-client] (ecmascript) <export default as Shield>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$chart$2d$column$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__BarChart3$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/chart-column.mjs [app-client] (ecmascript) <export default as BarChart3>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$target$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Target$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/target.mjs [app-client] (ecmascript) <export default as Target>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$award$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Award$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/award.mjs [app-client] (ecmascript) <export default as Award>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$pen$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Edit2$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/pen.mjs [app-client] (ecmascript) <export default as Edit2>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$check$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Check$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/check.mjs [app-client] (ecmascript) <export default as Check>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2d$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader2$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/loader-circle.mjs [app-client] (ecmascript) <export default as Loader2>");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$jobs$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/api/jobs.ts [app-client] (ecmascript)");
;
var _s = __turbopack_context__.k.signature();
"use client";
;
;
;
function ProfileModal({ isOpen, onClose }) {
    _s();
    const [reports, setReports] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])([]);
    const [loading, setLoading] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(true);
    const [isEditingName, setIsEditingName] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(false);
    const [userName, setUserName] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])('Head Coach');
    const [licenseId, setLicenseId] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])('PRO-992');
    const nameInputRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "ProfileModal.useEffect": ()=>{
            if (!isOpen) return;
            // Load from local storage
            const savedName = localStorage.getItem('gg-profile-name');
            if (savedName) {
                setUserName(savedName);
                setLicenseId(generateLicenseFrom(savedName));
            }
            // Fetch real stats
            async function fetchStats() {
                try {
                    setLoading(true);
                    const data = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$jobs$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["listReports"])();
                    setReports(data || []);
                } catch (error) {
                    console.error("Failed to load reports for profile:", error);
                } finally{
                    setLoading(false);
                }
            }
            fetchStats();
        }
    }["ProfileModal.useEffect"], [
        isOpen
    ]);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "ProfileModal.useEffect": ()=>{
            if (isEditingName && nameInputRef.current) {
                nameInputRef.current.focus();
            }
        }
    }["ProfileModal.useEffect"], [
        isEditingName
    ]);
    const generateLicenseFrom = (name)=>{
        const sum = name.split('').reduce((acc, char)=>acc + char.charCodeAt(0), 0);
        return `PRO-${sum % 10000}`;
    };
    const handleNameSave = ()=>{
        setIsEditingName(false);
        const finalName = userName.trim() || 'Head Coach';
        setUserName(finalName);
        localStorage.setItem('gg-profile-name', finalName);
        setLicenseId(generateLicenseFrom(finalName));
    };
    const handleKeyDown = (e)=>{
        if (e.key === 'Enter') handleNameSave();
        if (e.key === 'Escape') {
            setIsEditingName(false);
            setUserName(localStorage.getItem('gg-profile-name') || 'Head Coach');
        }
    };
    if (!isOpen) return null;
    // Compute stats
    const matchesAnalyzed = reports.length;
    const totalInsights = reports.reduce((sum, r)=>sum + (r.flaw_count || 0), 0);
    // Calculate level based on XP (say, 50 XP per match, 10 XP per insight)
    const currentXP = matchesAnalyzed * 50 + totalInsights * 10;
    // Give them a base level of 42 so they don't lose their "Pro" status if they have no matches, plus their calculated level
    const level = 42 + Math.floor(currentXP / 100);
    const xpIntoCurrentLevel = currentXP % 100;
    const xpForNextLevel = 100;
    const xpPercentage = xpIntoCurrentLevel / xpForNextLevel * 100;
    // Average accuracy based on win probability delta from 50 (just a flavor metric)
    let avgAccuracy = 98.2;
    if (reports.length > 0) {
        const total = reports.reduce((sum, r)=>{
            const wp = r.win_probability?.team_red || 50;
            return sum + (100 - Math.abs(50 - wp));
        }, 0);
        avgAccuracy = Number((total / reports.length).toFixed(1));
    }
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "fixed inset-0 z-[100] flex items-center justify-center bg-black/90 backdrop-blur-md font-sans p-4 animate-fade-in",
        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
            className: "bg-[#0a0f0a] border border-emerald-500/30 rounded-2xl w-full max-w-2xl shadow-[0_0_120px_rgba(16,185,129,0.15)] overflow-hidden relative",
            children: [
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: "absolute inset-0 z-0 opacity-20 pointer-events-none",
                    style: {
                        backgroundImage: 'radial-gradient(circle at 50% 0%, #10b981 0%, transparent 60%)'
                    }
                }, void 0, false, {
                    fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                    lineNumber: 95,
                    columnNumber: 9
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: "h-40 bg-gradient-to-r from-emerald-900/60 via-[#0a0f0a] to-blue-900/40 relative border-b border-emerald-500/20",
                    children: [
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "absolute inset-0 bg-grid opacity-30"
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                            lineNumber: 101,
                            columnNumber: 13
                        }, this),
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                            onClick: onClose,
                            className: "absolute top-4 right-4 text-gray-500 hover:text-white transition-colors z-10 bg-black/50 p-1.5 rounded-full backdrop-blur-sm",
                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$x$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__X$3e$__["X"], {
                                size: 20
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                lineNumber: 104,
                                columnNumber: 17
                            }, this)
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                            lineNumber: 103,
                            columnNumber: 13
                        }, this),
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "absolute -bottom-12 left-8 z-10 group",
                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "w-28 h-28 rounded-2xl bg-[#0a0f0a] border-2 border-emerald-500 flex items-center justify-center shadow-[0_0_30px_rgba(16,185,129,0.4)] relative overflow-hidden transition-transform duration-300 group-hover:scale-105",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "absolute inset-0 bg-emerald-500/10 animate-pulse"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                        lineNumber: 108,
                                        columnNumber: 21
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$user$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__User$3e$__["User"], {
                                        size: 56,
                                        className: "text-emerald-500 relative z-10"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                        lineNumber: 109,
                                        columnNumber: 21
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                lineNumber: 107,
                                columnNumber: 17
                            }, this)
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                            lineNumber: 106,
                            columnNumber: 13
                        }, this)
                    ]
                }, void 0, true, {
                    fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                    lineNumber: 99,
                    columnNumber: 9
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: "pt-16 px-8 pb-8 relative z-10",
                    children: [
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "flex justify-between items-start mb-8",
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "flex items-center gap-3 group",
                                            children: isEditingName ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: "flex items-center gap-2",
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                                        ref: nameInputRef,
                                                        type: "text",
                                                        value: userName,
                                                        onChange: (e)=>setUserName(e.target.value),
                                                        onKeyDown: handleKeyDown,
                                                        className: "bg-[#111a12] border border-emerald-500/50 text-gray-100 text-2xl font-bold rounded px-2 py-1 outline-none w-64 focus:ring-1 focus:ring-emerald-500",
                                                        placeholder: "Enter your name..."
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                        lineNumber: 121,
                                                        columnNumber: 33
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                        onClick: handleNameSave,
                                                        className: "p-1.5 bg-emerald-500/20 text-emerald-400 rounded hover:bg-emerald-500/40 transition-colors",
                                                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$check$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Check$3e$__["Check"], {
                                                            size: 18
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                            lineNumber: 131,
                                                            columnNumber: 37
                                                        }, this)
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                        lineNumber: 130,
                                                        columnNumber: 33
                                                    }, this)
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                lineNumber: 120,
                                                columnNumber: 29
                                            }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], {
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                                                        className: "text-2xl font-bold text-gray-100 flex items-center gap-2",
                                                        children: [
                                                            userName,
                                                            " ",
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$shield$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Shield$3e$__["Shield"], {
                                                                size: 18,
                                                                className: "text-emerald-500"
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                                lineNumber: 137,
                                                                columnNumber: 48
                                                            }, this)
                                                        ]
                                                    }, void 0, true, {
                                                        fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                        lineNumber: 136,
                                                        columnNumber: 33
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                        onClick: ()=>setIsEditingName(true),
                                                        className: "opacity-0 group-hover:opacity-100 p-1 text-gray-500 hover:text-emerald-400 transition-all",
                                                        title: "Edit Profile Name",
                                                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$pen$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Edit2$3e$__["Edit2"], {
                                                            size: 16
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                            lineNumber: 144,
                                                            columnNumber: 37
                                                        }, this)
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                        lineNumber: 139,
                                                        columnNumber: 33
                                                    }, this)
                                                ]
                                            }, void 0, true)
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                            lineNumber: 118,
                                            columnNumber: 21
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                            className: "text-emerald-500 font-mono text-xs uppercase tracking-widest mt-1",
                                            children: [
                                                "Tactical Analysis License: ",
                                                licenseId
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                            lineNumber: 149,
                                            columnNumber: 21
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                    lineNumber: 117,
                                    columnNumber: 17
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "text-right",
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "bg-emerald-500/10 border border-emerald-500/30 px-4 py-1.5 rounded-lg text-emerald-400 text-sm font-bold font-mono shadow-[0_0_15px_rgba(16,185,129,0.2)]",
                                            children: [
                                                "LEVEL ",
                                                level
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                            lineNumber: 153,
                                            columnNumber: 21
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "mt-2 flex items-center gap-2",
                                            title: `${xpIntoCurrentLevel} / ${xpForNextLevel} XP`,
                                            children: [
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                    className: "w-24 h-1.5 bg-gray-900 rounded-full overflow-hidden",
                                                    children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "h-full bg-emerald-500 shadow-[0_0_10px_#10b981] transition-all duration-1000 ease-out",
                                                        style: {
                                                            width: `${xpPercentage}%`
                                                        }
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                        lineNumber: 158,
                                                        columnNumber: 29
                                                    }, this)
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                    lineNumber: 157,
                                                    columnNumber: 25
                                                }, this),
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                    className: "text-[9px] font-mono text-gray-500",
                                                    children: [
                                                        xpIntoCurrentLevel,
                                                        " XP"
                                                    ]
                                                }, void 0, true, {
                                                    fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                    lineNumber: 163,
                                                    columnNumber: 25
                                                }, this)
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                            lineNumber: 156,
                                            columnNumber: 21
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                    lineNumber: 152,
                                    columnNumber: 17
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                            lineNumber: 116,
                            columnNumber: 13
                        }, this),
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "grid grid-cols-1 md:grid-cols-2 gap-4 mt-4",
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "bg-gradient-to-b from-[#111a12] to-[#0d140e] border border-gray-800/80 p-5 rounded-xl relative overflow-hidden group hover:border-emerald-500/30 transition-colors",
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity",
                                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$chart$2d$column$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__BarChart3$3e$__["BarChart3"], {
                                                size: 64
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                lineNumber: 172,
                                                columnNumber: 25
                                            }, this)
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                            lineNumber: 171,
                                            columnNumber: 21
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "flex items-center gap-2 text-emerald-500/70 text-[10px] font-bold uppercase tracking-widest mb-4",
                                            children: [
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$chart$2d$column$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__BarChart3$3e$__["BarChart3"], {
                                                    size: 14
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                    lineNumber: 175,
                                                    columnNumber: 25
                                                }, this),
                                                " Intelligence Stats"
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                            lineNumber: 174,
                                            columnNumber: 21
                                        }, this),
                                        loading ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "flex justify-center py-6",
                                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2d$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader2$3e$__["Loader2"], {
                                                className: "animate-spin text-emerald-500/50",
                                                size: 24
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                lineNumber: 180,
                                                columnNumber: 29
                                            }, this)
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                            lineNumber: 179,
                                            columnNumber: 25
                                        }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "space-y-4 relative z-10",
                                            children: [
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                    className: "flex justify-between items-end border-b border-gray-900/50 pb-2",
                                                    children: [
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                            className: "text-xs text-gray-400 font-mono uppercase",
                                                            children: "Matches Analyzed"
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                            lineNumber: 185,
                                                            columnNumber: 33
                                                        }, this),
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                            className: "text-lg font-bold text-gray-100",
                                                            children: matchesAnalyzed
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                            lineNumber: 186,
                                                            columnNumber: 33
                                                        }, this)
                                                    ]
                                                }, void 0, true, {
                                                    fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                    lineNumber: 184,
                                                    columnNumber: 29
                                                }, this),
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                    className: "flex justify-between items-end border-b border-gray-900/50 pb-2",
                                                    children: [
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                            className: "text-xs text-gray-400 font-mono uppercase",
                                                            children: "Total Insights"
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                            lineNumber: 189,
                                                            columnNumber: 33
                                                        }, this),
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                            className: "text-lg font-bold text-gray-100",
                                                            children: totalInsights
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                            lineNumber: 190,
                                                            columnNumber: 33
                                                        }, this)
                                                    ]
                                                }, void 0, true, {
                                                    fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                    lineNumber: 188,
                                                    columnNumber: 29
                                                }, this),
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                    className: "flex justify-between items-end",
                                                    children: [
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                            className: "text-xs text-gray-400 font-mono uppercase",
                                                            children: "Avg Accuracy"
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                            lineNumber: 193,
                                                            columnNumber: 33
                                                        }, this),
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                            className: "text-lg font-bold text-emerald-400",
                                                            children: [
                                                                avgAccuracy,
                                                                "%"
                                                            ]
                                                        }, void 0, true, {
                                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                            lineNumber: 194,
                                                            columnNumber: 33
                                                        }, this)
                                                    ]
                                                }, void 0, true, {
                                                    fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                    lineNumber: 192,
                                                    columnNumber: 29
                                                }, this)
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                            lineNumber: 183,
                                            columnNumber: 25
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                    lineNumber: 170,
                                    columnNumber: 17
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "bg-gradient-to-b from-[#111a12] to-[#0d140e] border border-gray-800/80 p-5 rounded-xl hover:border-emerald-500/30 transition-colors",
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "flex items-center gap-2 text-emerald-500/70 text-[10px] font-bold uppercase tracking-widest mb-4",
                                            children: [
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$target$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Target$3e$__["Target"], {
                                                    size: 14
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                    lineNumber: 203,
                                                    columnNumber: 25
                                                }, this),
                                                " Tactical Specialties"
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                            lineNumber: 202,
                                            columnNumber: 21
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "flex flex-wrap gap-2",
                                            children: [
                                                'Gegenpressing',
                                                'Low Block',
                                                'False 9',
                                                'Inverted Fullbacks'
                                            ].map((tag, i)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                    className: "px-3 py-1.5 bg-[#0a0f0a] text-gray-300 text-[11px] rounded border border-gray-800 hover:border-emerald-500/50 hover:text-emerald-400 hover:shadow-[0_0_10px_rgba(16,185,129,0.2)] transition-all cursor-default",
                                                    style: {
                                                        animationDelay: `${i * 100}ms`
                                                    },
                                                    children: tag
                                                }, tag, false, {
                                                    fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                    lineNumber: 207,
                                                    columnNumber: 29
                                                }, this))
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                            lineNumber: 205,
                                            columnNumber: 21
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "mt-4 pt-4 border-t border-gray-900/50",
                                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                                className: "text-[10px] text-gray-600 font-mono leading-relaxed",
                                                children: "Specialties are determined automatically by the AI engine based on your most frequent tactical analysis patterns."
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                                lineNumber: 217,
                                                columnNumber: 26
                                            }, this)
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                            lineNumber: 216,
                                            columnNumber: 21
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                    lineNumber: 201,
                                    columnNumber: 17
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                            lineNumber: 168,
                            columnNumber: 13
                        }, this),
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "mt-4 p-4 bg-gradient-to-r from-emerald-500/10 to-transparent border border-emerald-500/20 rounded-xl flex items-center gap-4 hover:border-emerald-500/40 transition-colors",
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "p-3 bg-[#0a0f0a] rounded-xl border border-emerald-500/30 shadow-[0_0_15px_rgba(16,185,129,0.2)]",
                                    children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$award$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Award$3e$__["Award"], {
                                        className: "text-emerald-500",
                                        size: 24
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                        lineNumber: 227,
                                        columnNumber: 21
                                    }, this)
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                    lineNumber: 226,
                                    columnNumber: 17
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h4", {
                                            className: "text-sm font-bold text-gray-200",
                                            children: "System Architect"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                            lineNumber: 230,
                                            columnNumber: 21
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                            className: "text-xs text-gray-400 mt-1",
                                            children: "Successfully integrated the Zero-Shot Learning branch."
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                            lineNumber: 231,
                                            columnNumber: 21
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                                    lineNumber: 229,
                                    columnNumber: 17
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                            lineNumber: 225,
                            columnNumber: 13
                        }, this)
                    ]
                }, void 0, true, {
                    fileName: "[project]/app/workspace/components/ProfileModal.tsx",
                    lineNumber: 115,
                    columnNumber: 9
                }, this)
            ]
        }, void 0, true, {
            fileName: "[project]/app/workspace/components/ProfileModal.tsx",
            lineNumber: 92,
            columnNumber: 7
        }, this)
    }, void 0, false, {
        fileName: "[project]/app/workspace/components/ProfileModal.tsx",
        lineNumber: 91,
        columnNumber: 5
    }, this);
}
_s(ProfileModal, "vO1LKzHWw8A1Yii4W0BwglmBtGY=");
_c = ProfileModal;
var _c;
__turbopack_context__.k.register(_c, "ProfileModal");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/app/workspace/components/Sidebar.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "Sidebar",
    ()=>Sidebar
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$layout$2d$dashboard$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__LayoutDashboard$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/layout-dashboard.mjs [app-client] (ecmascript) <export default as LayoutDashboard>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$file$2d$text$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__FileText$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/file-text.mjs [app-client] (ecmascript) <export default as FileText>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$settings$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Settings$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/settings.mjs [app-client] (ecmascript) <export default as Settings>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$server$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Server$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/server.mjs [app-client] (ecmascript) <export default as Server>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$menu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Menu$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/menu.mjs [app-client] (ecmascript) <export default as Menu>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$user$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__User$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/user.mjs [app-client] (ecmascript) <export default as User>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$users$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Users$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/users.mjs [app-client] (ecmascript) <export default as Users>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$book$2d$open$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__BookOpen$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/book-open.mjs [app-client] (ecmascript) <export default as BookOpen>");
var __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$EngineSettingsModal$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/app/workspace/components/EngineSettingsModal.tsx [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$ProfileModal$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/app/workspace/components/ProfileModal.tsx [app-client] (ecmascript)");
;
var _s = __turbopack_context__.k.signature();
"use client";
;
;
;
;
function Sidebar({ currentView = 'dashboard', setCurrentView = ()=>{} }) {
    _s();
    const [isEngineSettingsOpen, setIsEngineSettingsOpen] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(false);
    const [isProfileOpen, setIsProfileOpen] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(false);
    const [isCollapsed, setIsCollapsed] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(false);
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], {
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: `${isCollapsed ? 'w-[68px]' : 'w-[240px]'} flex-shrink-0 bg-[#0a0f0a] border-r border-gray-900 flex flex-col h-full font-mono relative z-10 transition-all duration-300 overflow-hidden`,
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex items-center justify-between p-4 border-b border-gray-900/50",
                        children: [
                            !isCollapsed && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "text-[10px] font-bold text-gray-500 uppercase tracking-widest pl-1 whitespace-nowrap",
                                children: "Workspace"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                lineNumber: 24,
                                columnNumber: 29
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                onClick: ()=>setIsCollapsed(!isCollapsed),
                                className: `p-1.5 rounded-md hover:bg-gray-800 text-gray-500 hover:text-emerald-400 transition-colors ${isCollapsed ? 'mx-auto' : ''}`,
                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$menu$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Menu$3e$__["Menu"], {
                                    size: 16
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                    lineNumber: 28,
                                    columnNumber: 14
                                }, this)
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                lineNumber: 25,
                                columnNumber: 12
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/Sidebar.tsx",
                        lineNumber: 23,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex-1 py-4",
                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("nav", {
                            className: "space-y-1",
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                    onClick: ()=>setCurrentView('dashboard'),
                                    title: "Dashboard",
                                    className: `w-full flex items-center gap-3 py-2.5 transition-colors ${currentView === 'dashboard' ? 'bg-[#111a12] text-gray-100 border-r-2 border-emerald-500' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-900/50'} ${isCollapsed ? 'justify-center px-0' : 'px-6'}`,
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$layout$2d$dashboard$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__LayoutDashboard$3e$__["LayoutDashboard"], {
                                            size: 18,
                                            className: currentView === 'dashboard' ? 'text-emerald-500' : ''
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                            lineNumber: 41,
                                            columnNumber: 15
                                        }, this),
                                        !isCollapsed && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                            className: `text-sm ${currentView === 'dashboard' ? 'font-semibold' : ''}`,
                                            children: "Dashboard"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                            lineNumber: 42,
                                            columnNumber: 32
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                    lineNumber: 35,
                                    columnNumber: 13
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                    onClick: ()=>setCurrentView('players'),
                                    title: "Player Reports",
                                    className: `w-full flex items-center gap-3 py-2.5 transition-colors ${currentView === 'players' ? 'bg-[#111a12] text-gray-100 border-r-2 border-emerald-500' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-900/50'} ${isCollapsed ? 'justify-center px-0' : 'px-6'}`,
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$users$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Users$3e$__["Users"], {
                                            size: 18,
                                            className: currentView === 'players' ? 'text-emerald-500' : ''
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                            lineNumber: 50,
                                            columnNumber: 15
                                        }, this),
                                        !isCollapsed && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                            className: `text-sm ${currentView === 'players' ? 'font-semibold' : ''}`,
                                            children: "Player Reports"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                            lineNumber: 51,
                                            columnNumber: 32
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                    lineNumber: 44,
                                    columnNumber: 13
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                    onClick: ()=>setCurrentView('dictionary'),
                                    title: "Dictionary",
                                    className: `w-full flex items-center gap-3 py-2.5 transition-colors ${currentView === 'dictionary' ? 'bg-[#111a12] text-gray-100 border-r-2 border-emerald-500' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-900/50'} ${isCollapsed ? 'justify-center px-0' : 'px-6'}`,
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$book$2d$open$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__BookOpen$3e$__["BookOpen"], {
                                            size: 18,
                                            className: currentView === 'dictionary' ? 'text-emerald-500' : ''
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                            lineNumber: 59,
                                            columnNumber: 15
                                        }, this),
                                        !isCollapsed && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                            className: `text-sm ${currentView === 'dictionary' ? 'font-semibold' : ''}`,
                                            children: "Dictionary"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                            lineNumber: 60,
                                            columnNumber: 32
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                    lineNumber: 53,
                                    columnNumber: 13
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                    onClick: ()=>setCurrentView('reports'),
                                    title: "Reports",
                                    className: `w-full flex items-center gap-3 py-2.5 transition-colors ${currentView === 'reports' ? 'bg-[#111a12] text-gray-100 border-r-2 border-emerald-500' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-900/50'} ${isCollapsed ? 'justify-center px-0' : 'px-6'}`,
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$file$2d$text$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__FileText$3e$__["FileText"], {
                                            size: 18,
                                            className: currentView === 'reports' ? 'text-emerald-500' : ''
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                            lineNumber: 68,
                                            columnNumber: 15
                                        }, this),
                                        !isCollapsed && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                            className: `text-sm ${currentView === 'reports' ? 'font-semibold' : ''}`,
                                            children: "Reports"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                            lineNumber: 69,
                                            columnNumber: 32
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                    lineNumber: 62,
                                    columnNumber: 13
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                    onClick: ()=>setIsProfileOpen(true),
                                    title: "Profile",
                                    className: `w-full flex items-center gap-3 py-2.5 text-gray-400 hover:text-gray-200 hover:bg-gray-900/50 transition-colors ${isCollapsed ? 'justify-center px-0' : 'px-6'}`,
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$user$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__User$3e$__["User"], {
                                            size: 18
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                            lineNumber: 75,
                                            columnNumber: 15
                                        }, this),
                                        !isCollapsed && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                            className: "text-sm",
                                            children: "Profile"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                            lineNumber: 76,
                                            columnNumber: 32
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                    lineNumber: 71,
                                    columnNumber: 13
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                    onClick: ()=>setIsEngineSettingsOpen(true),
                                    title: "Engine Settings",
                                    className: `w-full flex items-center gap-3 py-2.5 text-gray-400 hover:text-gray-200 hover:bg-gray-900/50 transition-colors ${isCollapsed ? 'justify-center px-0' : 'px-6'}`,
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$settings$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Settings$3e$__["Settings"], {
                                            size: 18
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                            lineNumber: 82,
                                            columnNumber: 15
                                        }, this),
                                        !isCollapsed && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                            className: "text-sm",
                                            children: "Engine Settings"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                            lineNumber: 83,
                                            columnNumber: 32
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                    lineNumber: 78,
                                    columnNumber: 13
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/app/workspace/components/Sidebar.tsx",
                            lineNumber: 34,
                            columnNumber: 11
                        }, this)
                    }, void 0, false, {
                        fileName: "[project]/app/workspace/components/Sidebar.tsx",
                        lineNumber: 33,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: `border-t border-gray-900 bg-[#0a0f0a] ${isCollapsed ? 'p-2 flex justify-center' : 'p-4'}`,
                        children: !isCollapsed ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], {
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "flex items-center justify-between text-xs mb-3",
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                            className: "text-gray-500",
                                            children: "Telemetry Engine"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                            lineNumber: 93,
                                            columnNumber: 17
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                            className: "text-emerald-500 font-bold flex items-center gap-1.5 bg-emerald-500/10 px-2 py-0.5 rounded",
                                            children: [
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$server$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Server$3e$__["Server"], {
                                                    size: 10
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                                    lineNumber: 95,
                                                    columnNumber: 19
                                                }, this),
                                                " LOCAL IDLE"
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                            lineNumber: 94,
                                            columnNumber: 17
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                    lineNumber: 92,
                                    columnNumber: 15
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "h-0.5 w-full bg-gray-800 overflow-hidden",
                                    children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "h-full bg-emerald-500 w-1/4"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                        lineNumber: 99,
                                        columnNumber: 17
                                    }, this)
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                    lineNumber: 98,
                                    columnNumber: 15
                                }, this)
                            ]
                        }, void 0, true) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            title: "Engine Status: LOCAL IDLE",
                            className: "my-2",
                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$server$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Server$3e$__["Server"], {
                                size: 18,
                                className: "text-emerald-500"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Sidebar.tsx",
                                lineNumber: 104,
                                columnNumber: 16
                            }, this)
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/components/Sidebar.tsx",
                            lineNumber: 103,
                            columnNumber: 13
                        }, this)
                    }, void 0, false, {
                        fileName: "[project]/app/workspace/components/Sidebar.tsx",
                        lineNumber: 89,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/Sidebar.tsx",
                lineNumber: 20,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$EngineSettingsModal$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["EngineSettingsModal"], {
                isOpen: isEngineSettingsOpen,
                onClose: ()=>setIsEngineSettingsOpen(false)
            }, void 0, false, {
                fileName: "[project]/app/workspace/components/Sidebar.tsx",
                lineNumber: 111,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$ProfileModal$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ProfileModal"], {
                isOpen: isProfileOpen,
                onClose: ()=>setIsProfileOpen(false)
            }, void 0, false, {
                fileName: "[project]/app/workspace/components/Sidebar.tsx",
                lineNumber: 115,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true);
}
_s(Sidebar, "f6D6Kd5knZKplvWTK01XjgiThqo=");
_c = Sidebar;
var _c;
__turbopack_context__.k.register(_c, "Sidebar");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/hooks/useWebSocketProgress.ts [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "STEPS",
    ()=>STEPS,
    "useWebSocketProgress",
    ()=>useWebSocketProgress
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/apiBase.ts [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$debugSessionLog$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/debugSessionLog.ts [app-client] (ecmascript)");
var _s = __turbopack_context__.k.signature();
;
;
;
const STEPS = [
    'Tracking Players',
    'Spatial Math',
    'Rule Engine',
    'Synthesizing Advice'
];
function useWebSocketProgress() {
    _s();
    const [currentStep, setCurrentStep] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])('Pending');
    const [isProcessing, setIsProcessing] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(false);
    const [error, setError] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(null);
    const wsRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    const pollingTimeoutRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    const isUnmounted = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(false);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "useWebSocketProgress.useEffect": ()=>{
            return ({
                "useWebSocketProgress.useEffect": ()=>{
                    isUnmounted.current = true;
                    if (pollingTimeoutRef.current) {
                        clearTimeout(pollingTimeoutRef.current);
                    }
                    if (wsRef.current) {
                        wsRef.current.close();
                    }
                }
            })["useWebSocketProgress.useEffect"];
        }
    }["useWebSocketProgress.useEffect"], []);
    const startTracking = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "useWebSocketProgress.useCallback[startTracking]": (jobId)=>{
            setIsProcessing(true);
            setError(null);
            setCurrentStep('Pending');
            isUnmounted.current = false;
            const connectWs = {
                "useWebSocketProgress.useCallback[startTracking].connectWs": (reconnectAttempts = 0)=>{
                    if (isUnmounted.current) return;
                    const wsUrl = `${(0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getWsBaseUrl"])()}/ws/jobs/${jobId}`;
                    const ws = new WebSocket(wsUrl);
                    wsRef.current = ws;
                    // #region agent log
                    (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$debugSessionLog$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["debugSessionLog"])({
                        sessionId: 'bb63ae',
                        hypothesisId: 'H2-H3',
                        location: 'useWebSocketProgress.ts:startTracking',
                        message: 'WS open url',
                        data: {
                            wsHost: ({
                                "useWebSocketProgress.useCallback[startTracking].connectWs": ()=>{
                                    try {
                                        return new URL(wsUrl).host;
                                    } catch  {
                                        return 'bad-ws-url';
                                    }
                                }
                            })["useWebSocketProgress.useCallback[startTracking].connectWs"](),
                            jobIdPrefix: jobId.slice(0, 8)
                        }
                    });
                    // #endregion
                    ws.onmessage = ({
                        "useWebSocketProgress.useCallback[startTracking].connectWs": (event)=>{
                            try {
                                const data = JSON.parse(event.data);
                                // #region agent log
                                (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$debugSessionLog$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["debugSessionLog"])({
                                    sessionId: 'bb63ae',
                                    hypothesisId: 'H2',
                                    location: 'useWebSocketProgress.ts:onmessage',
                                    message: 'WS payload',
                                    data: {
                                        status: data.status,
                                        current_step: data.current_step,
                                        hasError: !!data.error
                                    }
                                });
                                // #endregion
                                if (data.current_step != null && data.current_step !== '') {
                                    setCurrentStep(data.current_step);
                                }
                                const status = data.status;
                                if (status === 'done') {
                                    setCurrentStep('Completed');
                                    setIsProcessing(false);
                                    ws.close();
                                    return;
                                }
                                if (status === 'error') {
                                    const detail = typeof data.error === 'string' && data.error.length > 0 ? data.error : 'Job failed';
                                    setError(detail);
                                    setIsProcessing(false);
                                    ws.close();
                                }
                            } catch (e) {
                                console.error('Failed to parse WS message', e);
                            }
                        }
                    })["useWebSocketProgress.useCallback[startTracking].connectWs"];
                    ws.onerror = ({
                        "useWebSocketProgress.useCallback[startTracking].connectWs": (e)=>{
                            console.error('WebSocket Error', e);
                            // #region agent log
                            (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$debugSessionLog$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["debugSessionLog"])({
                                sessionId: 'bb63ae',
                                hypothesisId: 'H2-H3',
                                location: 'useWebSocketProgress.ts:onerror',
                                message: 'WS error',
                                data: {
                                    jobIdPrefix: jobId.slice(0, 8)
                                }
                            });
                        // #endregion
                        }
                    })["useWebSocketProgress.useCallback[startTracking].connectWs"];
                    ws.onclose = ({
                        "useWebSocketProgress.useCallback[startTracking].connectWs": ()=>{
                            if (isUnmounted.current) return;
                            setCurrentStep({
                                "useWebSocketProgress.useCallback[startTracking].connectWs": (prev)=>{
                                    if (prev === 'Completed') {
                                        setIsProcessing(false);
                                        return prev;
                                    }
                                    // Reconnect attempt if we haven't reached max retries
                                    if (reconnectAttempts < 3) {
                                        setTimeout({
                                            "useWebSocketProgress.useCallback[startTracking].connectWs": ()=>connectWs(reconnectAttempts + 1)
                                        }["useWebSocketProgress.useCallback[startTracking].connectWs"], 2000);
                                        return prev;
                                    }
                                    // Fallback to polling
                                    const pollStatus = {
                                        "useWebSocketProgress.useCallback[startTracking].connectWs.pollStatus": async ()=>{
                                            if (isUnmounted.current) return;
                                            try {
                                                const res = await fetch(`${(0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getWsBaseUrl"])().replace('ws://', 'http://').replace('wss://', 'https://')}/api/v1/jobs/${jobId}/artifacts`, {
                                                    headers: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
                                                });
                                                if (res.ok) {
                                                    const data = await res.json();
                                                    if (data.status === 'done') {
                                                        setCurrentStep('Completed');
                                                        setIsProcessing(false);
                                                        return;
                                                    } else if (data.status === 'error') {
                                                        setError('Job failed during processing.');
                                                        setIsProcessing(false);
                                                        return;
                                                    }
                                                } else if (res.status === 404) {
                                                    setError('job_not_found');
                                                    setIsProcessing(false);
                                                    return;
                                                }
                                            } catch (e) {
                                            // Ignore fetch errors, try again next tick
                                            }
                                            if (!isUnmounted.current) {
                                                pollingTimeoutRef.current = setTimeout(pollStatus, 2000);
                                            }
                                        }
                                    }["useWebSocketProgress.useCallback[startTracking].connectWs.pollStatus"];
                                    pollStatus();
                                    return prev;
                                }
                            }["useWebSocketProgress.useCallback[startTracking].connectWs"]);
                        }
                    })["useWebSocketProgress.useCallback[startTracking].connectWs"];
                }
            }["useWebSocketProgress.useCallback[startTracking].connectWs"];
            connectWs();
        }
    }["useWebSocketProgress.useCallback[startTracking]"], []);
    const disconnect = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "useWebSocketProgress.useCallback[disconnect]": ()=>{
            isUnmounted.current = true;
            if (wsRef.current) {
                wsRef.current.close();
            }
            if (pollingTimeoutRef.current) {
                clearTimeout(pollingTimeoutRef.current);
            }
            setIsProcessing(false);
        }
    }["useWebSocketProgress.useCallback[disconnect]"], []);
    return {
        currentStep,
        isProcessing,
        error,
        startTracking,
        disconnect
    };
}
_s(useWebSocketProgress, "FWv3z9DVO5uneaF7wz5pZ/iMT0s=");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/lib/api/chunkedUpload.ts [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "CHUNK_SIZE",
    ()=>CHUNK_SIZE,
    "completeUpload",
    ()=>completeUpload,
    "getUploadStatus",
    ()=>getUploadStatus,
    "initUpload",
    ()=>initUpload,
    "uploadChunk",
    ()=>uploadChunk
]);
/**
 * Low-level API client for the chunked upload protocol.
 *
 * Protocol:
 *   1. initUpload()     → POST /api/v1/upload/init
 *   2. uploadChunk()    → POST /api/v1/upload/chunk   (repeated per chunk)
 *   3. completeUpload() → POST /api/v1/upload/complete
 *   4. getUploadStatus()→ GET  /api/v1/upload/{id}/status  (for resume)
 */ var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/apiBase.ts [app-client] (ecmascript)");
;
const CHUNK_SIZE = 10 * 1024 * 1024;
async function initUpload(filename, totalSize, totalChunks) {
    const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
    const form = new FormData();
    form.append('filename', filename);
    form.append('total_size', totalSize.toString());
    form.append('total_chunks', totalChunks.toString());
    const res = await fetch(`${base}/api/v1/upload/init`, {
        method: 'POST',
        headers: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])(),
        body: form
    });
    if (!res.ok) {
        const detail = await res.text();
        throw new Error(`Failed to initialize upload: ${detail}`);
    }
    return res.json();
}
function uploadChunk(uploadId, chunkIndex, chunkBlob, signal, onChunkProgress) {
    return new Promise((resolve, reject)=>{
        const xhr = new XMLHttpRequest();
        const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
        xhr.open('POST', `${base}/api/v1/upload/chunk`);
        const headers = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])();
        for(const key in headers){
            xhr.setRequestHeader(key, headers[key]);
        }
        // Wire up abort signal
        if (signal) {
            if (signal.aborted) {
                reject(new DOMException('Upload cancelled', 'AbortError'));
                return;
            }
            signal.addEventListener('abort', ()=>{
                xhr.abort();
                reject(new DOMException('Upload cancelled', 'AbortError'));
            });
        }
        // Sub-chunk progress via XHR upload events
        if (onChunkProgress) {
            xhr.upload.addEventListener('progress', (e)=>{
                if (e.lengthComputable) {
                    onChunkProgress(e.loaded, e.total);
                }
            });
        }
        xhr.onload = ()=>{
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    resolve(JSON.parse(xhr.responseText));
                } catch  {
                    reject(new Error('Invalid JSON response from chunk upload'));
                }
            } else {
                reject(new Error(`Chunk upload failed (HTTP ${xhr.status}): ${xhr.responseText}`));
            }
        };
        xhr.onerror = ()=>reject(new Error('Network error during chunk upload'));
        xhr.ontimeout = ()=>reject(new Error('Chunk upload timed out'));
        // 5 minute timeout per chunk — extremely generous for 10 MB
        xhr.timeout = 5 * 60 * 1000;
        const form = new FormData();
        form.append('upload_id', uploadId);
        form.append('chunk_index', chunkIndex.toString());
        form.append('file', chunkBlob, `chunk_${chunkIndex}`);
        xhr.send(form);
    });
}
async function completeUpload(uploadId, opts) {
    const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
    const form = new FormData();
    form.append('upload_id', uploadId);
    form.append('cv_engine', opts.cvEngine);
    form.append('llm_engine', opts.llmEngine);
    form.append('quality_profile', opts.qualityProfile);
    form.append('chunking_interval', opts.chunkingInterval);
    const res = await fetch(`${base}/api/v1/upload/complete`, {
        method: 'POST',
        headers: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])(),
        body: form
    });
    if (!res.ok) {
        const detail = await res.text();
        throw new Error(`Failed to finalize upload: ${detail}`);
    }
    return res.json();
}
async function getUploadStatus(uploadId) {
    const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
    const res = await fetch(`${base}/api/v1/upload/${uploadId}/status`, {
        headers: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
    });
    if (!res.ok) {
        const detail = await res.text();
        throw new Error(`Failed to get upload status: ${detail}`);
    }
    return res.json();
}
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/hooks/useChunkedUpload.ts [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "useChunkedUpload",
    ()=>useChunkedUpload
]);
/**
 * React hook for chunked video uploads with progress, retry, cancel, and resume.
 *
 * Usage:
 *   const upload = useChunkedUpload();
 *   upload.startUpload(file, { cvEngine, llmEngine, ... });
 *   // render upload.progress, upload.phase, upload.speed, upload.timeRemaining
 */ var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$chunkedUpload$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/api/chunkedUpload.ts [app-client] (ecmascript)");
var _s = __turbopack_context__.k.signature();
;
;
// ── Constants ───────────────────────────────────────────────────────────
const MAX_RETRIES_PER_CHUNK = 3;
const RETRY_BASE_DELAY_MS = 1_000;
const SPEED_WINDOW = 5; // rolling average over last N chunks
// ── Helpers ─────────────────────────────────────────────────────────────
function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}
function formatDuration(seconds) {
    if (!isFinite(seconds) || seconds < 0) return '—';
    if (seconds < 60) return `~${Math.ceil(seconds)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.ceil(seconds % 60);
    if (m < 60) return `~${m}m ${s}s`;
    const h = Math.floor(m / 60);
    const rm = m % 60;
    return `~${h}h ${rm}m`;
}
function sleep(ms) {
    return new Promise((r)=>setTimeout(r, ms));
}
function useChunkedUpload() {
    _s();
    const [progress, setProgress] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(0);
    const [phase, setPhase] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])('idle');
    const [speed, setSpeed] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])('—');
    const [timeRemaining, setTimeRemaining] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])('—');
    const [chunksUploaded, setChunksUploaded] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(0);
    const [totalChunks, setTotalChunks] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(0);
    const [error, setError] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(null);
    // Refs for cancel/retry
    const abortRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    const lastFileRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    const lastOptsRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    const uploadIdRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    const startUpload = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "useChunkedUpload.useCallback[startUpload]": async (file, opts)=>{
            // Store for retry
            lastFileRef.current = file;
            lastOptsRef.current = opts;
            // Abort any previous upload
            abortRef.current?.abort();
            const controller = new AbortController();
            abortRef.current = controller;
            const numChunks = Math.ceil(file.size / __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$chunkedUpload$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["CHUNK_SIZE"]);
            setTotalChunks(numChunks);
            setChunksUploaded(0);
            setProgress(0);
            setSpeed('—');
            setTimeRemaining('—');
            setError(null);
            // ── Phase 1: Initialize ─────────────────────────────────────────
            setPhase('initializing');
            let uploadId;
            let alreadyReceived;
            // Check if we have a previous upload_id to resume
            const prevUploadId = uploadIdRef.current;
            if (prevUploadId) {
                try {
                    const status = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$chunkedUpload$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getUploadStatus"])(prevUploadId);
                    uploadId = prevUploadId;
                    alreadyReceived = new Set(status.received_chunks);
                    setChunksUploaded(alreadyReceived.size);
                    setProgress(alreadyReceived.size / numChunks * 100);
                } catch  {
                    // Previous session expired — start fresh
                    const initRes = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$chunkedUpload$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["initUpload"])(file.name, file.size, numChunks);
                    uploadId = initRes.upload_id;
                    alreadyReceived = new Set();
                }
            } else {
                const initRes = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$chunkedUpload$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["initUpload"])(file.name, file.size, numChunks);
                uploadId = initRes.upload_id;
                alreadyReceived = new Set();
            }
            uploadIdRef.current = uploadId;
            // ── Phase 2: Upload chunks ──────────────────────────────────────
            setPhase('uploading');
            const chunkTimesMs = []; // for speed estimation
            let uploaded = alreadyReceived.size;
            for(let i = 0; i < numChunks; i++){
                if (controller.signal.aborted) {
                    throw new DOMException('Upload cancelled', 'AbortError');
                }
                // Skip already-received chunks (resume)
                if (alreadyReceived.has(i)) continue;
                const start = i * __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$chunkedUpload$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["CHUNK_SIZE"];
                const end = Math.min(start + __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$chunkedUpload$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["CHUNK_SIZE"], file.size);
                const chunkBlob = file.slice(start, end);
                // Retry loop for this chunk
                let lastErr = null;
                for(let attempt = 0; attempt < MAX_RETRIES_PER_CHUNK; attempt++){
                    if (controller.signal.aborted) {
                        throw new DOMException('Upload cancelled', 'AbortError');
                    }
                    try {
                        const chunkStart = Date.now();
                        await (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$chunkedUpload$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["uploadChunk"])(uploadId, i, chunkBlob, controller.signal, {
                            "useChunkedUpload.useCallback[startUpload]": // Sub-chunk progress: interpolate between chunk boundaries
                            (loaded, total)=>{
                                const chunkFraction = total > 0 ? loaded / total : 0;
                                const overallProgress = (uploaded + chunkFraction) / numChunks * 100;
                                setProgress(Math.min(overallProgress, 99.9));
                            }
                        }["useChunkedUpload.useCallback[startUpload]"]);
                        const elapsed = Date.now() - chunkStart;
                        chunkTimesMs.push(elapsed);
                        // Update speed & ETA using rolling window
                        const recentTimes = chunkTimesMs.slice(-SPEED_WINDOW);
                        const avgTimeMs = recentTimes.reduce({
                            "useChunkedUpload.useCallback[startUpload]": (a, b)=>a + b
                        }["useChunkedUpload.useCallback[startUpload]"], 0) / recentTimes.length;
                        const bytesPerSec = chunkBlob.size / avgTimeMs * 1000;
                        setSpeed(`${formatBytes(bytesPerSec)}/s`);
                        const remainingChunks = numChunks - uploaded - 1;
                        const etaSec = remainingChunks * avgTimeMs / 1000;
                        setTimeRemaining(formatDuration(etaSec));
                        uploaded++;
                        setChunksUploaded(uploaded);
                        setProgress(uploaded / numChunks * 100);
                        lastErr = null;
                        break; // chunk succeeded
                    } catch (err) {
                        lastErr = err instanceof Error ? err : new Error(String(err));
                        if (lastErr.name === 'AbortError' || controller.signal.aborted) {
                            throw lastErr;
                        }
                        // Exponential backoff before retry
                        if (attempt < MAX_RETRIES_PER_CHUNK - 1) {
                            const delay = RETRY_BASE_DELAY_MS * Math.pow(2, attempt);
                            await sleep(delay);
                        }
                    }
                }
                if (lastErr) {
                    setPhase('error');
                    setError(`Failed to upload chunk ${i + 1}/${numChunks} after ${MAX_RETRIES_PER_CHUNK} attempts: ${lastErr.message}`);
                    throw lastErr;
                }
            }
            // ── Phase 3: Assemble & create job ──────────────────────────────
            setPhase('assembling');
            setProgress(100);
            setSpeed('—');
            setTimeRemaining('—');
            try {
                const result = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$chunkedUpload$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["completeUpload"])(uploadId, opts);
                setPhase('done');
                uploadIdRef.current = null; // session consumed
                return result;
            } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                setPhase('error');
                setError(`Failed to finalize upload: ${msg}`);
                throw err;
            }
        }
    }["useChunkedUpload.useCallback[startUpload]"], []);
    const cancelUpload = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "useChunkedUpload.useCallback[cancelUpload]": ()=>{
            abortRef.current?.abort();
            setPhase('idle');
            setProgress(0);
            setError(null);
            setSpeed('—');
            setTimeRemaining('—');
        // Keep uploadIdRef so we can resume if desired
        }
    }["useChunkedUpload.useCallback[cancelUpload]"], []);
    const retryUpload = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "useChunkedUpload.useCallback[retryUpload]": ()=>{
            const file = lastFileRef.current;
            const opts = lastOptsRef.current;
            if (!file || !opts) return;
            // startUpload will check uploadIdRef for resume
            void startUpload(file, opts);
        }
    }["useChunkedUpload.useCallback[retryUpload]"], [
        startUpload
    ]);
    const reset = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "useChunkedUpload.useCallback[reset]": ()=>{
            abortRef.current?.abort();
            setPhase('idle');
            setProgress(0);
            setError(null);
            setSpeed('—');
            setTimeRemaining('—');
            setChunksUploaded(0);
            setTotalChunks(0);
            uploadIdRef.current = null;
            lastFileRef.current = null;
            lastOptsRef.current = null;
        }
    }["useChunkedUpload.useCallback[reset]"], []);
    return {
        progress,
        phase,
        speed,
        timeRemaining,
        chunksUploaded,
        totalChunks,
        error,
        startUpload,
        cancelUpload,
        retryUpload,
        reset
    };
}
_s(useChunkedUpload, "n186kQW2hOAa7kf3a+0ROhthHuI=");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/app/workspace/components/Hopper.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "Hopper",
    ()=>Hopper
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$build$2f$polyfills$2f$process$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = /*#__PURE__*/ __turbopack_context__.i("[project]/node_modules/next/dist/build/polyfills/process.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$styled$2d$jsx$2f$style$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/styled-jsx/style.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useWebSocketProgress$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/hooks/useWebSocketProgress.ts [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useChunkedUpload$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/hooks/useChunkedUpload.ts [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$cloud$2d$upload$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__UploadCloud$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/cloud-upload.mjs [app-client] (ecmascript) <export default as UploadCloud>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$circle$2d$check$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__CheckCircle2$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/circle-check.mjs [app-client] (ecmascript) <export default as CheckCircle2>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2d$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader2$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/loader-circle.mjs [app-client] (ecmascript) <export default as Loader2>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Circle$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/circle.mjs [app-client] (ecmascript) <export default as Circle>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$circle$2d$alert$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__AlertCircle$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/circle-alert.mjs [app-client] (ecmascript) <export default as AlertCircle>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$circle$2d$x$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__XCircle$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/circle-x.mjs [app-client] (ecmascript) <export default as XCircle>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$refresh$2d$cw$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__RefreshCw$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/refresh-cw.mjs [app-client] (ecmascript) <export default as RefreshCw>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$hard$2d$drive$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__HardDrive$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/hard-drive.mjs [app-client] (ecmascript) <export default as HardDrive>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$wifi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Wifi$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/wifi.mjs [app-client] (ecmascript) <export default as Wifi>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$zap$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Zap$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/zap.mjs [app-client] (ecmascript) <export default as Zap>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$clock$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Clock$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/clock.mjs [app-client] (ecmascript) <export default as Clock>");
;
var _s = __turbopack_context__.k.signature();
;
;
;
;
;
function Hopper({ onComplete }) {
    _s();
    const { currentStep, isProcessing, error: wsError, startTracking } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useWebSocketProgress$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useWebSocketProgress"])();
    const chunked = (0, __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useChunkedUpload$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useChunkedUpload"])();
    const [file, setFile] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(null);
    const [activeJobId, setActiveJobId] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(null);
    const fileInputRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    // When pipeline processing completes, notify parent
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "Hopper.useEffect": ()=>{
            if (currentStep === 'Completed' && activeJobId && file) {
                onComplete(activeJobId, file);
            }
        }
    }["Hopper.useEffect"], [
        currentStep,
        activeJobId,
        file,
        onComplete
    ]);
    // Track milestone progress for the pipeline phase
    const stepIndexInMilestones = __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useWebSocketProgress$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["STEPS"].findIndex((s)=>currentStep === s || s === 'Tracking Players' && currentStep.toLowerCase().includes('tracking') || s === 'Spatial Math' && currentStep.toLowerCase().includes('spatial') || s === 'Rule Engine' && (currentStep.toLowerCase().includes('rule') || currentStep.toLowerCase().includes('tactical')) || s === 'Synthesizing Advice' && (currentStep.toLowerCase().includes('synth') || currentStep.toLowerCase().includes('llm')));
    const currentIndex = isProcessing || currentStep === 'Completed' ? stepIndexInMilestones >= 0 ? stepIndexInMilestones : currentStep.toLowerCase().includes('tracking') ? 0 : -1 : -1;
    if (currentStep === 'Completed') {
        return null;
    }
    const handleFileChange = async (e)=>{
        const selectedFile = e.target.files?.[0];
        if (!selectedFile) return;
        setFile(selectedFile);
        const llmPref = ("TURBOPACK compile-time value", "object") !== 'undefined' && localStorage.getItem('gaffer-engine-type') === 'cloud' ? 'cloud' : 'local';
        const cvPref = __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$build$2f$polyfills$2f$process$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"].env.NEXT_PUBLIC_CV_ENGINE === 'cloud' ? 'cloud' : 'local';
        const qualityPref = ("TURBOPACK compile-time truthy", 1) ? localStorage.getItem('gaffer-quality-profile') || 'balanced' : "TURBOPACK unreachable";
        const chunkingPref = ("TURBOPACK compile-time truthy", 1) ? localStorage.getItem('gaffer-chunking-interval') || '15-minute intervals' : "TURBOPACK unreachable";
        try {
            const result = await chunked.startUpload(selectedFile, {
                cvEngine: cvPref,
                llmEngine: llmPref,
                qualityProfile: qualityPref,
                chunkingInterval: chunkingPref
            });
            // Upload complete → job created → start WebSocket tracking
            setActiveJobId(result.job_id);
            startTracking(result.job_id);
        } catch (err) {
            if (err?.name === 'AbortError') return; // user cancelled
            console.error('Upload failed:', err);
        }
    };
    // Derived state
    const isUploading = chunked.phase !== 'idle' && chunked.phase !== 'done' && chunked.phase !== 'error';
    const showUploadProgress = chunked.phase === 'uploading' || chunked.phase === 'assembling' || chunked.phase === 'initializing';
    const showPipelineProgress = isProcessing && chunked.phase === 'done' && !wsError;
    const uploadError = chunked.phase === 'error' ? chunked.error : null;
    const pipelineError = wsError || null;
    const showError = !!(uploadError || pipelineError);
    const isIdle = chunked.phase === 'idle' && !isProcessing && !wsError;
    // Format file size for display
    const fileSizeDisplay = file ? file.size >= 1024 * 1024 * 1024 ? `${(file.size / (1024 * 1024 * 1024)).toFixed(2)} GB` : `${(file.size / (1024 * 1024)).toFixed(0)} MB` : '';
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "jsx-e60208c7c0e093ca" + " " + "flex flex-col gap-6 p-6 w-full max-w-4xl mx-auto h-full justify-center",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "jsx-e60208c7c0e093ca" + " " + "text-center mb-4",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h1", {
                        className: "jsx-e60208c7c0e093ca" + " " + "text-3xl font-bold font-sans tracking-tight text-gray-200",
                        children: "Initialize Analysis Engine"
                    }, void 0, false, {
                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                        lineNumber: 109,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                        className: "jsx-e60208c7c0e093ca" + " " + "text-gray-500 font-mono text-sm mt-2",
                        children: "Local-first telemetry. Connect high-fidelity match footage to generate insights."
                    }, void 0, false, {
                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                        lineNumber: 110,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/Hopper.tsx",
                lineNumber: 108,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                onClick: ()=>{
                    if (isIdle) fileInputRef.current?.click();
                },
                className: "jsx-e60208c7c0e093ca" + " " + `w-full h-48 border-2 border-dashed rounded-xl flex flex-col items-center justify-center transition-all ${isIdle ? 'border-gray-700 bg-[#0a0f0a] hover:border-emerald-500/70 hover:bg-[#111a12]/50 cursor-pointer' : 'border-emerald-500/50 bg-[#111a12] shadow-[0_0_30px_rgba(16,185,129,0.05)]'}`,
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                        type: "file",
                        accept: "video/mp4,video/quicktime,video/x-msvideo",
                        ref: fileInputRef,
                        onChange: handleFileChange,
                        className: "jsx-e60208c7c0e093ca" + " " + "hidden"
                    }, void 0, false, {
                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                        lineNumber: 122,
                        columnNumber: 9
                    }, this),
                    isIdle ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$cloud$2d$upload$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__UploadCloud$3e$__["UploadCloud"], {
                                size: 40,
                                className: "text-gray-600 mb-3"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 132,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                                className: "jsx-e60208c7c0e093ca" + " " + "text-lg font-bold text-gray-300 tracking-tight font-sans",
                                children: "System Ready"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 133,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                className: "jsx-e60208c7c0e093ca" + " " + "text-xs text-gray-500 mt-1 font-mono",
                                children: "Click here to upload match footage"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 134,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                className: "jsx-e60208c7c0e093ca" + " " + "text-[10px] text-gray-600 mt-2 font-mono",
                                children: "Supports files up to 12 GB · Chunked upload with auto-resume"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 135,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true) : showUploadProgress ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "jsx-e60208c7c0e093ca" + " " + "text-emerald-500 mb-3",
                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$cloud$2d$upload$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__UploadCloud$3e$__["UploadCloud"], {
                                    size: 40,
                                    className: "animate-pulse"
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/Hopper.tsx",
                                    lineNumber: 140,
                                    columnNumber: 15
                                }, this)
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 139,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                                className: "jsx-e60208c7c0e093ca" + " " + "text-lg font-bold text-emerald-400 tracking-tight font-mono uppercase",
                                children: chunked.phase === 'initializing' ? 'Initializing Upload...' : chunked.phase === 'assembling' ? 'Assembling Video...' : 'Uploading Video...'
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 142,
                                columnNumber: 13
                            }, this),
                            file && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                className: "jsx-e60208c7c0e093ca" + " " + "text-[10px] text-gray-500 mt-1 font-mono",
                                children: [
                                    file.name,
                                    " · ",
                                    fileSizeDisplay
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 148,
                                columnNumber: 15
                            }, this)
                        ]
                    }, void 0, true) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "jsx-e60208c7c0e093ca" + " " + "text-emerald-500 mb-3",
                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2d$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader2$3e$__["Loader2"], {
                                    size: 40,
                                    className: "animate-spin"
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/Hopper.tsx",
                                    lineNumber: 154,
                                    columnNumber: 15
                                }, this)
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 153,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                                className: "jsx-e60208c7c0e093ca" + " " + "text-lg font-bold text-emerald-400 tracking-tight font-mono uppercase",
                                children: "Processing Pipeline..."
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 156,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/Hopper.tsx",
                lineNumber: 114,
                columnNumber: 7
            }, this),
            showUploadProgress && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "jsx-e60208c7c0e093ca" + " " + "bg-[#111a12] border border-[#1a2420] rounded-xl p-6 shadow-2xl animate-fade-in-up",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "jsx-e60208c7c0e093ca" + " " + "flex items-center justify-between mb-4",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h3", {
                                className: "jsx-e60208c7c0e093ca" + " " + "text-xs font-bold text-gray-500 uppercase tracking-widest font-mono",
                                children: "Upload Progress"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 167,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                onClick: ()=>chunked.cancelUpload(),
                                className: "jsx-e60208c7c0e093ca" + " " + "flex items-center gap-1.5 text-[10px] font-mono text-red-400/70 hover:text-red-400 transition-colors uppercase tracking-wider",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$circle$2d$x$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__XCircle$3e$__["XCircle"], {
                                        size: 12
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                                        lineNumber: 174,
                                        columnNumber: 15
                                    }, this),
                                    "Cancel"
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 170,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                        lineNumber: 166,
                        columnNumber: 11
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "jsx-e60208c7c0e093ca" + " " + "relative w-full h-3 bg-gray-800/80 rounded-full overflow-hidden mb-4",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                style: {
                                    width: `${Math.max(chunked.progress, 0.5)}%`,
                                    background: chunked.phase === 'assembling' ? 'linear-gradient(90deg, #059669, #10b981, #34d399)' : 'linear-gradient(90deg, #065f46, #059669, #10b981)'
                                },
                                className: "jsx-e60208c7c0e093ca" + " " + "absolute inset-y-0 left-0 rounded-full transition-all duration-300 ease-out"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 181,
                                columnNumber: 13
                            }, this),
                            chunked.phase === 'uploading' && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                style: {
                                    width: `${chunked.progress}%`,
                                    background: 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%)',
                                    animation: 'shimmer 2s ease-in-out infinite'
                                },
                                className: "jsx-e60208c7c0e093ca" + " " + "absolute inset-y-0 left-0 rounded-full opacity-30"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 192,
                                columnNumber: 15
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                        lineNumber: 180,
                        columnNumber: 11
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "jsx-e60208c7c0e093ca" + " " + "flex items-center justify-between text-[11px] font-mono",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "jsx-e60208c7c0e093ca" + " " + "flex items-center gap-4",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "jsx-e60208c7c0e093ca" + " " + "text-emerald-400 font-bold text-sm tabular-nums",
                                        children: [
                                            chunked.progress.toFixed(1),
                                            "%"
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                                        lineNumber: 207,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "jsx-e60208c7c0e093ca" + " " + "flex items-center gap-1 text-gray-500",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$hard$2d$drive$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__HardDrive$3e$__["HardDrive"], {
                                                size: 11
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                                lineNumber: 213,
                                                columnNumber: 17
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "jsx-e60208c7c0e093ca" + " " + "tabular-nums",
                                                children: [
                                                    chunked.chunksUploaded,
                                                    "/",
                                                    chunked.totalChunks
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                                lineNumber: 214,
                                                columnNumber: 17
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "jsx-e60208c7c0e093ca" + " " + "text-gray-700",
                                                children: "chunks"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                                lineNumber: 215,
                                                columnNumber: 17
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                                        lineNumber: 212,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 205,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "jsx-e60208c7c0e093ca" + " " + "flex items-center gap-4",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "jsx-e60208c7c0e093ca" + " " + "flex items-center gap-1 text-gray-500",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$wifi$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Wifi$3e$__["Wifi"], {
                                                size: 11
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                                lineNumber: 222,
                                                columnNumber: 17
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "jsx-e60208c7c0e093ca" + " " + "tabular-nums",
                                                children: chunked.speed
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                                lineNumber: 223,
                                                columnNumber: 17
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                                        lineNumber: 221,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "jsx-e60208c7c0e093ca" + " " + "flex items-center gap-1 text-gray-500",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$clock$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Clock$3e$__["Clock"], {
                                                size: 11
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                                lineNumber: 228,
                                                columnNumber: 17
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "jsx-e60208c7c0e093ca" + " " + "tabular-nums",
                                                children: chunked.timeRemaining
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                                lineNumber: 229,
                                                columnNumber: 17
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                                        lineNumber: 227,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 219,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                        lineNumber: 204,
                        columnNumber: 11
                    }, this),
                    chunked.phase === 'assembling' && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "jsx-e60208c7c0e093ca" + " " + "flex items-center gap-2 mt-4 text-emerald-400 text-xs font-mono",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$zap$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Zap$3e$__["Zap"], {
                                size: 14,
                                className: "animate-pulse"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 237,
                                columnNumber: 15
                            }, this),
                            "Reassembling video on server... This takes a moment for large files."
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                        lineNumber: 236,
                        columnNumber: 13
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/Hopper.tsx",
                lineNumber: 165,
                columnNumber: 9
            }, this),
            showError && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "jsx-e60208c7c0e093ca" + " " + "bg-[#1a1212] border border-red-900/40 rounded-xl p-5 shadow-2xl",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "jsx-e60208c7c0e093ca" + " " + "flex items-start gap-3",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$circle$2d$alert$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__AlertCircle$3e$__["AlertCircle"], {
                                size: 18,
                                className: "text-red-400 flex-shrink-0 mt-0.5"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 248,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "jsx-e60208c7c0e093ca" + " " + "flex-1 min-w-0",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "jsx-e60208c7c0e093ca" + " " + "text-red-400 font-mono text-sm font-bold mb-1",
                                        children: uploadError ? 'Upload Failed' : 'Pipeline Error'
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                                        lineNumber: 250,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "jsx-e60208c7c0e093ca" + " " + "text-red-300/70 font-mono text-xs break-words",
                                        children: uploadError || pipelineError
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                                        lineNumber: 253,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 249,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                        lineNumber: 247,
                        columnNumber: 11
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "jsx-e60208c7c0e093ca" + " " + "flex items-center gap-3 mt-4",
                        children: [
                            uploadError && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                onClick: ()=>chunked.retryUpload(),
                                className: "jsx-e60208c7c0e093ca" + " " + "flex items-center gap-1.5 px-4 py-2 bg-emerald-600/20 hover:bg-emerald-600/30 border border-emerald-600/40 rounded-lg text-emerald-400 text-xs font-mono font-bold uppercase tracking-wider transition-all",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$refresh$2d$cw$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__RefreshCw$3e$__["RefreshCw"], {
                                        size: 13
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                                        lineNumber: 264,
                                        columnNumber: 17
                                    }, this),
                                    "Retry Upload"
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 260,
                                columnNumber: 15
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                onClick: ()=>{
                                    chunked.reset();
                                    setFile(null);
                                    setActiveJobId(null);
                                },
                                className: "jsx-e60208c7c0e093ca" + " " + "flex items-center gap-1.5 px-4 py-2 bg-gray-800/40 hover:bg-gray-800/60 border border-gray-700/40 rounded-lg text-gray-400 text-xs font-mono uppercase tracking-wider transition-all",
                                children: "Start Over"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 268,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                        lineNumber: 258,
                        columnNumber: 11
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/Hopper.tsx",
                lineNumber: 246,
                columnNumber: 9
            }, this),
            isIdle && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "jsx-e60208c7c0e093ca" + " " + "w-full flex flex-col sm:flex-row items-center justify-center gap-6 animate-fade-in-up mt-2",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "jsx-e60208c7c0e093ca" + " " + "flex items-center gap-3",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                className: "jsx-e60208c7c0e093ca" + " " + "text-[10px] font-bold text-gray-500 uppercase tracking-widest font-mono",
                                children: "Quality Profile"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 287,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("select", {
                                defaultValue: ("TURBOPACK compile-time truthy", 1) ? localStorage.getItem('gaffer-quality-profile') || 'balanced' : "TURBOPACK unreachable",
                                onChange: (e)=>{
                                    if ("TURBOPACK compile-time truthy", 1) localStorage.setItem('gaffer-quality-profile', e.target.value);
                                },
                                className: "jsx-e60208c7c0e093ca" + " " + "bg-[#111a12] border border-gray-800 rounded-lg px-4 py-2 text-gray-300 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500/50 transition-all font-mono text-xs appearance-none cursor-pointer",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("option", {
                                        value: "fast",
                                        className: "jsx-e60208c7c0e093ca",
                                        children: "Fast (Preview)"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                                        lineNumber: 295,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("option", {
                                        value: "balanced",
                                        className: "jsx-e60208c7c0e093ca",
                                        children: "Balanced (Standard)"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                                        lineNumber: 296,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("option", {
                                        value: "high_res",
                                        className: "jsx-e60208c7c0e093ca",
                                        children: "High Res (Detailed)"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                                        lineNumber: 297,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("option", {
                                        value: "sahi",
                                        className: "jsx-e60208c7c0e093ca",
                                        children: "SAHI (Maximum Ball Recall)"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                                        lineNumber: 298,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 288,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                        lineNumber: 286,
                        columnNumber: 11
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "jsx-e60208c7c0e093ca" + " " + "flex items-center gap-3",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("label", {
                                className: "jsx-e60208c7c0e093ca" + " " + "text-[10px] font-bold text-gray-500 uppercase tracking-widest font-mono",
                                children: "Chunking Interval"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 303,
                                columnNumber: 13
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("select", {
                                defaultValue: ("TURBOPACK compile-time truthy", 1) ? localStorage.getItem('gaffer-chunking-interval') || '15-minute intervals' : "TURBOPACK unreachable",
                                onChange: (e)=>{
                                    if ("TURBOPACK compile-time truthy", 1) localStorage.setItem('gaffer-chunking-interval', e.target.value);
                                },
                                className: "jsx-e60208c7c0e093ca" + " " + "bg-[#111a12] border border-gray-800 rounded-lg px-4 py-2 text-gray-300 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500/50 transition-all font-mono text-xs appearance-none cursor-pointer",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("option", {
                                        value: "15-minute intervals",
                                        className: "jsx-e60208c7c0e093ca",
                                        children: "15-minute intervals"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                                        lineNumber: 311,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("option", {
                                        value: "30-minute intervals",
                                        className: "jsx-e60208c7c0e093ca",
                                        children: "30-minute intervals"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                                        lineNumber: 312,
                                        columnNumber: 15
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("option", {
                                        value: "Full Halves (45 mins)",
                                        className: "jsx-e60208c7c0e093ca",
                                        children: "Full Halves (45 mins)"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                                        lineNumber: 313,
                                        columnNumber: 15
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 304,
                                columnNumber: 13
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                        lineNumber: 302,
                        columnNumber: 11
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/Hopper.tsx",
                lineNumber: 284,
                columnNumber: 9
            }, this),
            showPipelineProgress && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "jsx-e60208c7c0e093ca" + " " + "bg-[#111a12] border border-[#1a2420] rounded-xl p-6 shadow-2xl mt-4",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h3", {
                        className: "jsx-e60208c7c0e093ca" + " " + "text-xs font-bold text-gray-500 uppercase tracking-widest mb-3 border-b border-gray-800 pb-3",
                        children: "Telemetry Pipeline Tracker"
                    }, void 0, false, {
                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                        lineNumber: 323,
                        columnNumber: 11
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                        className: "jsx-e60208c7c0e093ca" + " " + "text-[11px] font-mono text-emerald-300/90 mb-4",
                        children: [
                            "Backend step: ",
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                className: "jsx-e60208c7c0e093ca" + " " + "text-gray-100",
                                children: currentStep
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 325,
                                columnNumber: 27
                            }, this),
                            stepIndexInMilestones === -1 && currentStep !== 'Pending' && currentStep !== 'Completed' ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                className: "jsx-e60208c7c0e093ca" + " " + "ml-2 text-gray-500",
                                children: "(not in milestone list)"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 327,
                                columnNumber: 15
                            }, this) : null
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                        lineNumber: 324,
                        columnNumber: 11
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "jsx-e60208c7c0e093ca" + " " + "flex flex-col gap-5",
                        children: __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useWebSocketProgress$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["STEPS"].map((step, idx)=>{
                            const matchesCurrent = step === currentStep;
                            const isPast = currentStep === 'Completed' || currentIndex >= 0 && idx < currentIndex;
                            return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "jsx-e60208c7c0e093ca" + " " + "flex items-center gap-4",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "jsx-e60208c7c0e093ca" + " " + "flex-shrink-0",
                                        children: isPast ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$circle$2d$check$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__CheckCircle2$3e$__["CheckCircle2"], {
                                            size: 18,
                                            className: "text-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.3)] rounded-full"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/Hopper.tsx",
                                            lineNumber: 341,
                                            columnNumber: 23
                                        }, this) : matchesCurrent ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2d$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader2$3e$__["Loader2"], {
                                            size: 18,
                                            className: "text-emerald-400 animate-spin"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/Hopper.tsx",
                                            lineNumber: 343,
                                            columnNumber: 23
                                        }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Circle$3e$__["Circle"], {
                                            size: 18,
                                            className: "text-gray-700"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/Hopper.tsx",
                                            lineNumber: 345,
                                            columnNumber: 23
                                        }, this)
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                                        lineNumber: 339,
                                        columnNumber: 19
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "jsx-e60208c7c0e093ca" + " " + `font-mono text-sm tracking-wide ${isPast ? 'text-gray-400' : matchesCurrent ? 'text-emerald-400 font-bold' : 'text-gray-700'}`,
                                        children: step
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                                        lineNumber: 348,
                                        columnNumber: 19
                                    }, this)
                                ]
                            }, step, true, {
                                fileName: "[project]/app/workspace/components/Hopper.tsx",
                                lineNumber: 338,
                                columnNumber: 17
                            }, this);
                        })
                    }, void 0, false, {
                        fileName: "[project]/app/workspace/components/Hopper.tsx",
                        lineNumber: 330,
                        columnNumber: 11
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/Hopper.tsx",
                lineNumber: 322,
                columnNumber: 9
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$styled$2d$jsx$2f$style$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"], {
                id: "e60208c7c0e093ca",
                children: "@keyframes shimmer{0%{transform:translate(-100%)}to{transform:translate(200%)}}"
            }, void 0, false, void 0, this)
        ]
    }, void 0, true, {
        fileName: "[project]/app/workspace/components/Hopper.tsx",
        lineNumber: 105,
        columnNumber: 5
    }, this);
}
_s(Hopper, "ujweu/kj57yXQ/7uTvjz18JBC0k=", false, function() {
    return [
        __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useWebSocketProgress$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useWebSocketProgress"],
        __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useChunkedUpload$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useChunkedUpload"]
    ];
});
_c = Hopper;
var _c;
__turbopack_context__.k.register(_c, "Hopper");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/app/workspace/components/VideoHUD.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>VideoHUD
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
;
var _s = __turbopack_context__.k.signature();
;
function formatClock(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
    const s = Math.floor(seconds);
    const h = Math.floor(s / 3600);
    const m = Math.floor(s % 3600 / 60);
    const sec = s % 60;
    if (h > 0) {
        return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
    }
    return `${m}:${String(sec).padStart(2, "0")}`;
}
function VideoHUD({ videoRef, videoSrc, jobId, status, onDownload, trackingData, useAltNames = false, dictionary = {} }) {
    _s();
    const [isPlaying, setIsPlaying] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(false);
    const [currentTime, setCurrentTime] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(0);
    const [duration, setDuration] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(0);
    const [showOverlays, setShowOverlays] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(true);
    const containerRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    const [isFullscreen, setIsFullscreen] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(false);
    const [geometry, setGeometry] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])({
        width: 0,
        height: 0,
        left: 0,
        top: 0,
        scaleX: 1,
        scaleY: 1,
        videoWidth: 1920,
        videoHeight: 1080
    });
    const updateGeometry = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "VideoHUD.useCallback[updateGeometry]": ()=>{
            const video = videoRef.current;
            if (!video) return;
            const rect = video.getBoundingClientRect();
            const videoW = video.videoWidth || 1920;
            const videoH = video.videoHeight || 1080;
            const containerRatio = rect.width / rect.height;
            const videoRatio = videoW / videoH;
            let displayedW = rect.width;
            let displayedH = rect.height;
            let left = 0;
            let top = 0;
            if (containerRatio > videoRatio) {
                displayedW = rect.height * videoRatio;
                left = (rect.width - displayedW) / 2;
            } else {
                displayedH = rect.width / videoRatio;
                top = (rect.height - displayedH) / 2;
            }
            setGeometry({
                width: displayedW,
                height: displayedH,
                left: left,
                top: top,
                scaleX: displayedW / videoW,
                scaleY: displayedH / videoH,
                videoWidth: videoW,
                videoHeight: videoH
            });
        }
    }["VideoHUD.useCallback[updateGeometry]"], [
        videoRef
    ]);
    // Keep geometry updated on resize or load
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "VideoHUD.useEffect": ()=>{
            const video = videoRef.current;
            if (!video) return;
            const observer = new ResizeObserver({
                "VideoHUD.useEffect": ()=>{
                    updateGeometry();
                }
            }["VideoHUD.useEffect"]);
            observer.observe(video);
            video.addEventListener("loadedmetadata", updateGeometry);
            return ({
                "VideoHUD.useEffect": ()=>{
                    observer.disconnect();
                    video.removeEventListener("loadedmetadata", updateGeometry);
                }
            })["VideoHUD.useEffect"];
        }
    }["VideoHUD.useEffect"], [
        videoRef,
        videoSrc,
        updateGeometry
    ]);
    const toggleFullscreen = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "VideoHUD.useCallback[toggleFullscreen]": ()=>{
            const container = containerRef.current;
            if (!container) return;
            if (!document.fullscreenElement) {
                container.requestFullscreen?.().catch({
                    "VideoHUD.useCallback[toggleFullscreen]": (err)=>{
                        console.error(`Error attempting to enable fullscreen: ${err.message}`);
                    }
                }["VideoHUD.useCallback[toggleFullscreen]"]);
            } else {
                document.exitFullscreen?.();
            }
        }
    }["VideoHUD.useCallback[toggleFullscreen]"], []);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "VideoHUD.useEffect": ()=>{
            const onFullscreenChange = {
                "VideoHUD.useEffect.onFullscreenChange": ()=>{
                    const isFull = document.fullscreenElement === containerRef.current;
                    setIsFullscreen(isFull);
                    updateGeometry();
                    setTimeout(updateGeometry, 50);
                    setTimeout(updateGeometry, 150);
                    setTimeout(updateGeometry, 300);
                }
            }["VideoHUD.useEffect.onFullscreenChange"];
            document.addEventListener("fullscreenchange", onFullscreenChange);
            document.addEventListener("webkitfullscreenchange", onFullscreenChange);
            document.addEventListener("mozfullscreenchange", onFullscreenChange);
            document.addEventListener("MSFullscreenChange", onFullscreenChange);
            return ({
                "VideoHUD.useEffect": ()=>{
                    document.removeEventListener("fullscreenchange", onFullscreenChange);
                    document.removeEventListener("webkitfullscreenchange", onFullscreenChange);
                    document.removeEventListener("mozfullscreenchange", onFullscreenChange);
                    document.removeEventListener("MSFullscreenChange", onFullscreenChange);
                }
            })["VideoHUD.useEffect"];
        }
    }["VideoHUD.useEffect"], [
        updateGeometry
    ]);
    const activeFrame = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "VideoHUD.useMemo[activeFrame]": ()=>{
            if (!trackingData?.frames || trackingData.frames.length === 0) return null;
            const fps = trackingData.telemetry?.fps || 25; // default to 25
            const targetIdx = Math.round(currentTime * fps);
            // Binary search for the closest frame index
            let closestFrame = trackingData.frames[0];
            let minDiff = Math.abs(closestFrame.frame_idx - targetIdx);
            let low = 0;
            let high = trackingData.frames.length - 1;
            while(low <= high){
                const mid = Math.floor((low + high) / 2);
                const diff = Math.abs(trackingData.frames[mid].frame_idx - targetIdx);
                if (diff < minDiff) {
                    minDiff = diff;
                    closestFrame = trackingData.frames[mid];
                }
                if (trackingData.frames[mid].frame_idx < targetIdx) {
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }
            return closestFrame;
        }
    }["VideoHUD.useMemo[activeFrame]"], [
        trackingData,
        currentTime
    ]);
    const filteredPlayers = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "VideoHUD.useMemo[filteredPlayers]": ()=>{
            if (!activeFrame?.players) return [];
            return activeFrame.players.filter({
                "VideoHUD.useMemo[filteredPlayers]": (p)=>p.id !== 75 && p.id !== 76 && p.id !== 79
            }["VideoHUD.useMemo[filteredPlayers]"]);
        }
    }["VideoHUD.useMemo[filteredPlayers]"], [
        activeFrame
    ]);
    const backLineDefenders = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "VideoHUD.useMemo[backLineDefenders]": ()=>{
            if (filteredPlayers.length === 0) return {
                team0: [],
                team1: []
            };
            const team0 = filteredPlayers.filter({
                "VideoHUD.useMemo[backLineDefenders].team0": (p)=>p.team_id === 'team_0' || p.team === 'team_0'
            }["VideoHUD.useMemo[backLineDefenders].team0"]);
            const team1 = filteredPlayers.filter({
                "VideoHUD.useMemo[backLineDefenders].team1": (p)=>p.team_id === 'team_1' || p.team === 'team_1'
            }["VideoHUD.useMemo[backLineDefenders].team1"]);
            const team0Sorted = [
                ...team0
            ].filter({
                "VideoHUD.useMemo[backLineDefenders].team0Sorted": (p)=>p.x_pitch !== null && p.y_pitch !== null
            }["VideoHUD.useMemo[backLineDefenders].team0Sorted"]).sort({
                "VideoHUD.useMemo[backLineDefenders].team0Sorted": (a, b)=>(a.x_pitch ?? 0) - (b.x_pitch ?? 0)
            }["VideoHUD.useMemo[backLineDefenders].team0Sorted"]);
            const team1Sorted = [
                ...team1
            ].filter({
                "VideoHUD.useMemo[backLineDefenders].team1Sorted": (p)=>p.x_pitch !== null && p.y_pitch !== null
            }["VideoHUD.useMemo[backLineDefenders].team1Sorted"]).sort({
                "VideoHUD.useMemo[backLineDefenders].team1Sorted": (a, b)=>(b.x_pitch ?? 0) - (a.x_pitch ?? 0)
            }["VideoHUD.useMemo[backLineDefenders].team1Sorted"]);
            // Skip goalkeeper (index 0) and take next 4
            const defenders0 = team0Sorted.slice(1, 5);
            const defenders1 = team1Sorted.slice(1, 5);
            // Sort by y_pitch to draw a line connecting them from top to bottom
            defenders0.sort({
                "VideoHUD.useMemo[backLineDefenders]": (a, b)=>(a.y_pitch ?? 0) - (b.y_pitch ?? 0)
            }["VideoHUD.useMemo[backLineDefenders]"]);
            defenders1.sort({
                "VideoHUD.useMemo[backLineDefenders]": (a, b)=>(a.y_pitch ?? 0) - (b.y_pitch ?? 0)
            }["VideoHUD.useMemo[backLineDefenders]"]);
            return {
                team0: defenders0,
                team1: defenders1
            };
        }
    }["VideoHUD.useMemo[backLineDefenders]"], [
        filteredPlayers
    ]);
    const getPitchDistance = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "VideoHUD.useCallback[getPitchDistance]": (p1, p2)=>{
            let dx = p1[0] - p2[0];
            let dy = p1[1] - p2[1];
            if (Math.abs(p1[0]) > 110 || Math.abs(p2[0]) > 110) {
                dx = dx / 10;
                dy = dy / 10;
            }
            return Math.hypot(dx, dy);
        }
    }["VideoHUD.useCallback[getPitchDistance]"], []);
    const ballArrowPath = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "VideoHUD.useMemo[ballArrowPath]": ()=>{
            if (!trackingData?.frames || !activeFrame) return null;
            const currentIdx = activeFrame.frame_idx;
            const activeFrameIndex = trackingData.frames.findIndex({
                "VideoHUD.useMemo[ballArrowPath].activeFrameIndex": (f)=>f.frame_idx === currentIdx
            }["VideoHUD.useMemo[ballArrowPath].activeFrameIndex"]);
            if (activeFrameIndex === -1) return null;
            const startIndex = Math.max(0, activeFrameIndex - 30);
            const pathPoints = [];
            for(let i = startIndex; i <= activeFrameIndex; i++){
                const f = trackingData.frames[i];
                if (f.ball_canvas && Array.isArray(f.ball_canvas) && f.ball_canvas.length >= 2) {
                    const [bx, by] = f.ball_canvas;
                    if (bx !== null && by !== null) {
                        pathPoints.push({
                            x: bx * geometry.scaleX,
                            y: by * geometry.scaleY,
                            frame_idx: f.frame_idx,
                            ball_xy: f.ball_xy
                        });
                    }
                }
            }
            return pathPoints;
        }
    }["VideoHUD.useMemo[ballArrowPath]"], [
        trackingData,
        activeFrame,
        geometry.scaleX,
        geometry.scaleY
    ]);
    const passArrow = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "VideoHUD.useMemo[passArrow]": ()=>{
            if (!ballArrowPath || ballArrowPath.length < 5) return null;
            const startPt = ballArrowPath[0];
            const endPt = ballArrowPath[ballArrowPath.length - 1];
            const canvasDist = Math.hypot(endPt.x - startPt.x, endPt.y - startPt.y);
            if (canvasDist < 40) return null;
            const midX = (startPt.x + endPt.x) / 2;
            const midY = (startPt.y + endPt.y) / 2;
            const heightOffset = Math.min(120, canvasDist * 0.35);
            const ctrlX = midX;
            const ctrlY = midY - heightOffset;
            const peakX = 0.25 * startPt.x + 0.5 * ctrlX + 0.25 * endPt.x;
            const peakY = 0.25 * startPt.y + 0.5 * ctrlY + 0.25 * endPt.y;
            const angle = Math.atan2(endPt.y - ctrlY, endPt.x - ctrlX);
            const arrowLength = 12;
            const arrowAngle = Math.PI / 6;
            const xLeft = endPt.x - arrowLength * Math.cos(angle - arrowAngle);
            const yLeft = endPt.y - arrowLength * Math.sin(angle - arrowAngle);
            const xRight = endPt.x - arrowLength * Math.cos(angle + arrowAngle);
            const yRight = endPt.y - arrowLength * Math.sin(angle + arrowAngle);
            let distanceMeters = 0;
            if (startPt.ball_xy && endPt.ball_xy) {
                distanceMeters = getPitchDistance(startPt.ball_xy, endPt.ball_xy);
            } else {
                distanceMeters = canvasDist * 0.15;
            }
            return {
                d: `M ${startPt.x} ${startPt.y} Q ${ctrlX} ${ctrlY} ${endPt.x} ${endPt.y}`,
                arrowPoints: `${endPt.x},${endPt.y} ${xLeft},${yLeft} ${xRight},${yRight}`,
                peak: {
                    x: peakX,
                    y: peakY
                },
                distance: distanceMeters
            };
        }
    }["VideoHUD.useMemo[passArrow]"], [
        ballArrowPath,
        getPitchDistance
    ]);
    const tacticalTriangles = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "VideoHUD.useMemo[tacticalTriangles]": ()=>{
            if (filteredPlayers.length === 0) return [];
            const teams = [
                'team_0',
                'team_1'
            ];
            const results = [];
            teams.forEach({
                "VideoHUD.useMemo[tacticalTriangles]": (teamId)=>{
                    const teamPlayers = filteredPlayers.filter({
                        "VideoHUD.useMemo[tacticalTriangles].teamPlayers": (p)=>(p.team_id === teamId || p.team === teamId) && p.x_pitch != null && p.y_pitch != null
                    }["VideoHUD.useMemo[tacticalTriangles].teamPlayers"]);
                    const n = teamPlayers.length;
                    const validTriplets = [];
                    for(let i = 0; i < n; i++){
                        for(let j = i + 1; j < n; j++){
                            for(let k = j + 1; k < n; k++){
                                const pA = teamPlayers[i];
                                const pB = teamPlayers[j];
                                const pC = teamPlayers[k];
                                const d_AB = getPitchDistance([
                                    pA.x_pitch,
                                    pA.y_pitch
                                ], [
                                    pB.x_pitch,
                                    pB.y_pitch
                                ]);
                                const d_BC = getPitchDistance([
                                    pB.x_pitch,
                                    pB.y_pitch
                                ], [
                                    pC.x_pitch,
                                    pC.y_pitch
                                ]);
                                const d_CA = getPitchDistance([
                                    pC.x_pitch,
                                    pC.y_pitch
                                ], [
                                    pA.x_pitch,
                                    pA.y_pitch
                                ]);
                                const minDist = 5;
                                const maxDist = 22;
                                if (d_AB >= minDist && d_AB <= maxDist && d_BC >= minDist && d_BC <= maxDist && d_CA >= minDist && d_CA <= maxDist) {
                                    const ptA = {
                                        x: pA.x_canvas * geometry.scaleX,
                                        y: pA.bbox ? pA.bbox[3] * geometry.scaleY : (pA.y_canvas + 15) * geometry.scaleY
                                    };
                                    const ptB = {
                                        x: pB.x_canvas * geometry.scaleX,
                                        y: pB.bbox ? pB.bbox[3] * geometry.scaleY : (pB.y_canvas + 15) * geometry.scaleY
                                    };
                                    const ptC = {
                                        x: pC.x_canvas * geometry.scaleX,
                                        y: pC.bbox ? pC.bbox[3] * geometry.scaleY : (pC.y_canvas + 15) * geometry.scaleY
                                    };
                                    validTriplets.push({
                                        points: [
                                            ptA,
                                            ptB,
                                            ptC
                                        ],
                                        perimeter: d_AB + d_BC + d_CA
                                    });
                                }
                            }
                        }
                    }
                    validTriplets.sort({
                        "VideoHUD.useMemo[tacticalTriangles]": (a, b)=>a.perimeter - b.perimeter
                    }["VideoHUD.useMemo[tacticalTriangles]"]);
                    const color = teamId === 'team_0' ? '#ef4444' : '#3b82f6';
                    const countToTake = Math.min(2, validTriplets.length);
                    for(let idx = 0; idx < countToTake; idx++){
                        const triplet = validTriplets[idx];
                        const ptsStr = triplet.points.map({
                            "VideoHUD.useMemo[tacticalTriangles].ptsStr": (p)=>`${p.x},${p.y}`
                        }["VideoHUD.useMemo[tacticalTriangles].ptsStr"]).join(' ');
                        results.push({
                            pointsStr: ptsStr,
                            color,
                            perimeter: triplet.perimeter
                        });
                    }
                }
            }["VideoHUD.useMemo[tacticalTriangles]"]);
            return results;
        }
    }["VideoHUD.useMemo[tacticalTriangles]"], [
        filteredPlayers,
        geometry,
        getPitchDistance
    ]);
    const detectedClips = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "VideoHUD.useMemo[detectedClips]": ()=>{
            if (!trackingData?.frames || trackingData.frames.length === 0) {
                return [
                    {
                        id: 'pass-demo',
                        title: 'Passing Arc',
                        desc: '3D Passing Trajectory Arc',
                        time: 9,
                        type: 'pass'
                    },
                    {
                        id: 'tri-demo',
                        title: 'Support Triangle',
                        desc: 'Teammate Passing Network',
                        time: 13,
                        type: 'triangle'
                    },
                    {
                        id: 'switch-demo',
                        title: 'Flank Switch',
                        desc: 'Long Curved Flank Switch',
                        time: 38,
                        type: 'pass'
                    },
                    {
                        id: 'press-demo',
                        title: 'Defensive Press',
                        desc: 'Defensive Line & Free Man',
                        time: 71,
                        type: 'defense'
                    }
                ];
            }
            const clips = [];
            const frames = trackingData.frames;
            const fps = trackingData.telemetry?.fps || 25;
            const step = Math.max(1, Math.round(fps));
            let foundPass = false;
            let foundTriangle = false;
            for(let i = 0; i < frames.length; i += step){
                const f = frames[i];
                const timeSec = f.frame_idx / fps;
                const team0Players = f.players.filter({
                    "VideoHUD.useMemo[detectedClips].team0Players": (p)=>p.team_id === 'team_0' && p.x_pitch != null && p.y_pitch != null
                }["VideoHUD.useMemo[detectedClips].team0Players"]);
                const team1Players = f.players.filter({
                    "VideoHUD.useMemo[detectedClips].team1Players": (p)=>p.team_id === 'team_1' && p.x_pitch != null && p.y_pitch != null
                }["VideoHUD.useMemo[detectedClips].team1Players"]);
                const hasTri = {
                    "VideoHUD.useMemo[detectedClips].hasTri": (players)=>{
                        const n = players.length;
                        for(let a = 0; a < n; a++){
                            for(let b = a + 1; b < n; b++){
                                for(let c = b + 1; c < n; c++){
                                    const d1 = getPitchDistance([
                                        players[a].x_pitch,
                                        players[a].y_pitch
                                    ], [
                                        players[b].x_pitch,
                                        players[b].y_pitch
                                    ]);
                                    const d2 = getPitchDistance([
                                        players[b].x_pitch,
                                        players[b].y_pitch
                                    ], [
                                        players[c].x_pitch,
                                        players[c].y_pitch
                                    ]);
                                    const d3 = getPitchDistance([
                                        players[c].x_pitch,
                                        players[c].y_pitch
                                    ], [
                                        players[a].x_pitch,
                                        players[a].y_pitch
                                    ]);
                                    if (d1 >= 5 && d1 <= 22 && d2 >= 5 && d2 <= 22 && d3 >= 5 && d3 <= 22) {
                                        return true;
                                    }
                                }
                            }
                        }
                        return false;
                    }
                }["VideoHUD.useMemo[detectedClips].hasTri"];
                if (!foundTriangle && (hasTri(team0Players) || hasTri(team1Players))) {
                    clips.push({
                        id: `tri-${f.frame_idx}`,
                        title: 'Support Network',
                        desc: 'Active teammate support triangle',
                        time: Math.round(timeSec),
                        type: 'triangle'
                    });
                    foundTriangle = true;
                }
                if (!foundPass && f.ball_canvas) {
                    const lookbackIdx = Math.max(0, i - 15);
                    const prevF = frames[lookbackIdx];
                    if (prevF.ball_canvas) {
                        const dist = Math.hypot(f.ball_canvas[0] - prevF.ball_canvas[0], f.ball_canvas[1] - prevF.ball_canvas[1]);
                        if (dist > 80) {
                            clips.push({
                                id: `pass-${f.frame_idx}`,
                                title: 'Passing Play',
                                desc: 'Ball trajectory arc detected',
                                time: Math.round(timeSec),
                                type: 'pass'
                            });
                            foundPass = true;
                        }
                    }
                }
                if (clips.length >= 6) break;
            }
            if (clips.length < 3) {
                return [
                    {
                        id: 'pass-demo',
                        title: 'Passing Arc',
                        desc: '3D Passing Trajectory Arc',
                        time: 9,
                        type: 'pass'
                    },
                    {
                        id: 'tri-demo',
                        title: 'Support Triangle',
                        desc: 'Teammate Passing Network',
                        time: 13,
                        type: 'triangle'
                    },
                    {
                        id: 'switch-demo',
                        title: 'Flank Switch',
                        desc: 'Long Curved Flank Switch',
                        time: 38,
                        type: 'pass'
                    },
                    {
                        id: 'press-demo',
                        title: 'Defensive Press',
                        desc: 'Defensive Line & Free Man',
                        time: 71,
                        type: 'defense'
                    }
                ];
            }
            return clips;
        }
    }["VideoHUD.useMemo[detectedClips]"], [
        trackingData,
        getPitchDistance
    ]);
    const playClip = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "VideoHUD.useCallback[playClip]": (timeSec)=>{
            const v = videoRef.current;
            if (!v) return;
            v.currentTime = timeSec;
            void v.play().catch({
                "VideoHUD.useCallback[playClip]": (err)=>console.log('Autoplay blocked or failed:', err)
            }["VideoHUD.useCallback[playClip]"]);
        }
    }["VideoHUD.useCallback[playClip]"], [
        videoRef
    ]);
    const getPlayerSpeed = (player, pIdx)=>{
        if (player.speed_kmh !== undefined && player.speed_kmh !== null) {
            return player.speed_kmh;
        }
        if (!trackingData?.frames || !player.id) return 0;
        const currentFrameIdx = activeFrame?.frame_idx;
        if (currentFrameIdx === undefined) return 0;
        const lookback = 5;
        const targetIdx = currentFrameIdx - lookback;
        const prevFrame = trackingData.frames.find((f)=>f.frame_idx === targetIdx);
        if (!prevFrame) return 0;
        const prevPlayer = prevFrame.players.find((p)=>p.id === player.id);
        if (!prevPlayer || prevPlayer.x_pitch === undefined || prevPlayer.y_pitch === undefined) return 0;
        const dx = player.x_pitch - prevPlayer.x_pitch;
        const dy = player.y_pitch - prevPlayer.y_pitch;
        const dist = Math.hypot(dx, dy);
        const fps = trackingData.telemetry?.fps || 25;
        const timeSec = lookback / fps;
        const speedMs = dist / timeSec;
        const speedKmh = speedMs * 3.6;
        return isNaN(speedKmh) ? 0 : Math.min(36, speedKmh);
    };
    const getTeamColor = (teamId)=>{
        if (teamId === 'team_0') return '#ef4444'; // Red
        if (teamId === 'team_1') return '#3b82f6'; // Blue
        return '#a1a1aa';
    };
    const syncFromVideo = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useCallback"])({
        "VideoHUD.useCallback[syncFromVideo]": ()=>{
            const v = videoRef.current;
            if (!v) return;
            const d = v.duration;
            setDuration(Number.isFinite(d) ? d : 0);
            setCurrentTime(Number.isFinite(v.currentTime) ? v.currentTime : 0);
            setIsPlaying(!v.paused);
        }
    }["VideoHUD.useCallback[syncFromVideo]"], [
        videoRef
    ]);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "VideoHUD.useEffect": ()=>{
            const v = videoRef.current;
            if (!v || !videoSrc) {
                setCurrentTime(0);
                setDuration(0);
                setIsPlaying(false);
                return;
            }
            const onLoadedMetadata = {
                "VideoHUD.useEffect.onLoadedMetadata": ()=>syncFromVideo()
            }["VideoHUD.useEffect.onLoadedMetadata"];
            const onTimeUpdate = {
                "VideoHUD.useEffect.onTimeUpdate": ()=>syncFromVideo()
            }["VideoHUD.useEffect.onTimeUpdate"];
            const onSeeked = {
                "VideoHUD.useEffect.onSeeked": ()=>syncFromVideo()
            }["VideoHUD.useEffect.onSeeked"];
            const onPlay = {
                "VideoHUD.useEffect.onPlay": ()=>setIsPlaying(true)
            }["VideoHUD.useEffect.onPlay"];
            const onPause = {
                "VideoHUD.useEffect.onPause": ()=>setIsPlaying(false)
            }["VideoHUD.useEffect.onPause"];
            const onEnded = {
                "VideoHUD.useEffect.onEnded": ()=>{
                    setIsPlaying(false);
                    syncFromVideo();
                }
            }["VideoHUD.useEffect.onEnded"];
            v.addEventListener("loadedmetadata", onLoadedMetadata);
            v.addEventListener("durationchange", onLoadedMetadata);
            v.addEventListener("timeupdate", onTimeUpdate);
            v.addEventListener("seeked", onSeeked);
            v.addEventListener("play", onPlay);
            v.addEventListener("pause", onPause);
            v.addEventListener("ended", onEnded);
            if (v.readyState >= HTMLMediaElement.HAVE_METADATA) {
                syncFromVideo();
            }
            return ({
                "VideoHUD.useEffect": ()=>{
                    v.removeEventListener("loadedmetadata", onLoadedMetadata);
                    v.removeEventListener("durationchange", onLoadedMetadata);
                    v.removeEventListener("timeupdate", onTimeUpdate);
                    v.removeEventListener("seeked", onSeeked);
                    v.removeEventListener("play", onPlay);
                    v.removeEventListener("pause", onPause);
                    v.removeEventListener("ended", onEnded);
                }
            })["VideoHUD.useEffect"];
        }
    }["VideoHUD.useEffect"], [
        videoRef,
        videoSrc,
        syncFromVideo
    ]);
    const togglePlay = ()=>{
        const v = videoRef.current;
        if (!v) return;
        if (v.paused) {
            void v.play();
        } else {
            v.pause();
        }
    };
    const onSeekInput = (e)=>{
        const v = videoRef.current;
        if (!v) return;
        const t = parseFloat(e.target.value);
        if (!Number.isFinite(t)) return;
        v.currentTime = t;
        setCurrentTime(t);
    };
    const onVideoAreaClick = ()=>{
        togglePlay();
    };
    const maxT = Number.isFinite(duration) && duration > 0 ? duration : 0;
    const rangeValue = Number.isFinite(currentTime) ? Math.min(Math.max(currentTime, 0), maxT || 0) : 0;
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "flex h-full min-h-0 min-w-0 flex-1 flex-col gap-2",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                ref: containerRef,
                className: `flex flex-1 min-h-0 flex-col overflow-hidden bg-black shadow-2xl relative ${isFullscreen ? "w-screen h-screen rounded-none border-0" : "rounded-xl border border-gray-800"}`,
                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: "relative flex min-h-0 flex-1 flex-col",
                    children: [
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "absolute left-3 top-3 z-10 flex gap-2",
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                    className: "rounded bg-black/60 px-2 py-1 text-xs font-bold text-white backdrop-blur-md",
                                    children: "LIVE FEED"
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                    lineNumber: 569,
                                    columnNumber: 13
                                }, this),
                                jobId && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                    className: "rounded bg-blue-900/80 px-2 py-1 text-xs font-mono text-blue-100 backdrop-blur-md",
                                    children: [
                                        jobId.slice(0, 8),
                                        "…"
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                    lineNumber: 573,
                                    columnNumber: 15
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                            lineNumber: 568,
                            columnNumber: 11
                        }, this),
                        videoSrc ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], {
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "relative min-h-0 flex-1 cursor-pointer bg-black flex items-center justify-center",
                                    onClick: onVideoAreaClick,
                                    onKeyDown: (e)=>{
                                        if (e.key === " " || e.key === "Enter") {
                                            e.preventDefault();
                                            togglePlay();
                                        }
                                    },
                                    role: "button",
                                    tabIndex: 0,
                                    "aria-label": "Play or pause video",
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("video", {
                                            ref: videoRef,
                                            src: videoSrc,
                                            className: "pointer-events-none h-full w-full object-contain",
                                            playsInline: true,
                                            preload: "metadata"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                            lineNumber: 594,
                                            columnNumber: 17
                                        }, this),
                                        showOverlays && activeFrame && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                                            className: "absolute pointer-events-none select-none z-10",
                                            style: {
                                                width: geometry.width,
                                                height: geometry.height,
                                                left: geometry.left,
                                                top: geometry.top
                                            },
                                            viewBox: `0 0 ${geometry.width} ${geometry.height}`,
                                            children: [
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("defs", {
                                                    children: [
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("filter", {
                                                            id: "shadow",
                                                            x: "-20%",
                                                            y: "-20%",
                                                            width: "140%",
                                                            height: "140%",
                                                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("feDropShadow", {
                                                                dx: "0",
                                                                dy: "2",
                                                                stdDeviation: "3",
                                                                floodOpacity: "0.6",
                                                                floodColor: "#000000"
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                lineNumber: 617,
                                                                columnNumber: 25
                                                            }, this)
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                            lineNumber: 616,
                                                            columnNumber: 23
                                                        }, this),
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("linearGradient", {
                                                            id: "pass-grad",
                                                            x1: "0%",
                                                            y1: "0%",
                                                            x2: "100%",
                                                            y2: "0%",
                                                            children: [
                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("stop", {
                                                                    offset: "0%",
                                                                    stopColor: "#ffffff",
                                                                    stopOpacity: "0.4"
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                    lineNumber: 620,
                                                                    columnNumber: 25
                                                                }, this),
                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("stop", {
                                                                    offset: "50%",
                                                                    stopColor: "#22c55e",
                                                                    stopOpacity: "0.9"
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                    lineNumber: 621,
                                                                    columnNumber: 25
                                                                }, this),
                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("stop", {
                                                                    offset: "100%",
                                                                    stopColor: "#22c55e",
                                                                    stopOpacity: "1"
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                    lineNumber: 622,
                                                                    columnNumber: 25
                                                                }, this)
                                                            ]
                                                        }, void 0, true, {
                                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                            lineNumber: 619,
                                                            columnNumber: 23
                                                        }, this)
                                                    ]
                                                }, void 0, true, {
                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                    lineNumber: 615,
                                                    columnNumber: 21
                                                }, this),
                                                tacticalTriangles.map((tri, tIdx)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("polygon", {
                                                        points: tri.pointsStr,
                                                        fill: `${tri.color}22`,
                                                        stroke: tri.color,
                                                        strokeWidth: 1.5,
                                                        strokeDasharray: "4,3",
                                                        opacity: 0.75,
                                                        style: {
                                                            filter: "url(#shadow)"
                                                        }
                                                    }, `support-tri-${tIdx}`, false, {
                                                        fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                        lineNumber: 628,
                                                        columnNumber: 23
                                                    }, this)),
                                                passArrow && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("g", {
                                                    style: {
                                                        filter: "url(#shadow)"
                                                    },
                                                    children: [
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                                            d: passArrow.d,
                                                            fill: "none",
                                                            stroke: "rgba(0, 0, 0, 0.4)",
                                                            strokeWidth: 5,
                                                            strokeLinecap: "round"
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                            lineNumber: 643,
                                                            columnNumber: 25
                                                        }, this),
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                                            d: passArrow.d,
                                                            fill: "none",
                                                            stroke: "url(#pass-grad)",
                                                            strokeWidth: 3,
                                                            strokeLinecap: "round"
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                            lineNumber: 650,
                                                            columnNumber: 25
                                                        }, this),
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("polygon", {
                                                            points: passArrow.arrowPoints,
                                                            fill: "#22c55e",
                                                            stroke: "white",
                                                            strokeWidth: 1
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                            lineNumber: 657,
                                                            columnNumber: 25
                                                        }, this),
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("g", {
                                                            transform: `translate(${passArrow.peak.x}, ${passArrow.peak.y})`,
                                                            children: [
                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("rect", {
                                                                    x: -28,
                                                                    y: -14,
                                                                    width: 56,
                                                                    height: 15,
                                                                    rx: 4,
                                                                    fill: "rgba(10, 15, 10, 0.85)",
                                                                    stroke: "#22c55e",
                                                                    strokeWidth: 1
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                    lineNumber: 664,
                                                                    columnNumber: 27
                                                                }, this),
                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("text", {
                                                                    x: 0,
                                                                    y: -3,
                                                                    textAnchor: "middle",
                                                                    fill: "#ffffff",
                                                                    fontSize: "9px",
                                                                    fontWeight: "extrabold",
                                                                    children: `${passArrow.distance.toFixed(1)} m`
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                    lineNumber: 674,
                                                                    columnNumber: 27
                                                                }, this)
                                                            ]
                                                        }, void 0, true, {
                                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                            lineNumber: 663,
                                                            columnNumber: 25
                                                        }, this)
                                                    ]
                                                }, "ball-pass-arrow", true, {
                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                    lineNumber: 642,
                                                    columnNumber: 23
                                                }, this),
                                                (()=>{
                                                    const { team0, team1 } = backLineDefenders;
                                                    const renders = [];
                                                    [
                                                        {
                                                            defenders: team0,
                                                            color: '#ef4444',
                                                            label: `${useAltNames && dictionary["team_0"] ? dictionary["team_0"] : "Red Team"} Line`
                                                        },
                                                        {
                                                            defenders: team1,
                                                            color: '#3b82f6',
                                                            label: `${useAltNames && dictionary["team_1"] ? dictionary["team_1"] : "Blue Team"} Line`
                                                        }
                                                    ].forEach(({ defenders, color, label }, idx)=>{
                                                        if (defenders.length < 2) return;
                                                        const pts = defenders.map((p)=>{
                                                            const x = p.x_canvas * geometry.scaleX;
                                                            const y = (p.bbox ? p.bbox[3] : p.y_canvas + 15) * geometry.scaleY;
                                                            return {
                                                                x,
                                                                y
                                                            };
                                                        });
                                                        const pointsStr = pts.map((p)=>`${p.x},${p.y}`).join(' ');
                                                        // Calculate center for text label
                                                        const cx = pts.reduce((sum, p)=>sum + p.x, 0) / pts.length;
                                                        const cy = pts.reduce((sum, p)=>sum + p.y, 0) / pts.length;
                                                        renders.push(/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("g", {
                                                            children: [
                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("polyline", {
                                                                    points: pointsStr,
                                                                    fill: "none",
                                                                    stroke: color,
                                                                    strokeWidth: 2,
                                                                    strokeDasharray: "6,4",
                                                                    opacity: 0.8
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                    lineNumber: 713,
                                                                    columnNumber: 29
                                                                }, this),
                                                                pts.map((p, pIdx)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("circle", {
                                                                        cx: p.x,
                                                                        cy: p.y,
                                                                        r: 4,
                                                                        fill: color,
                                                                        opacity: 0.9
                                                                    }, `joint-${pIdx}`, false, {
                                                                        fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                        lineNumber: 723,
                                                                        columnNumber: 31
                                                                    }, this)),
                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("rect", {
                                                                    x: cx - 40,
                                                                    y: cy - 20,
                                                                    width: 80,
                                                                    height: 13,
                                                                    rx: 3,
                                                                    fill: "rgba(0, 0, 0, 0.75)",
                                                                    stroke: color,
                                                                    strokeWidth: 1
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                    lineNumber: 733,
                                                                    columnNumber: 29
                                                                }, this),
                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("text", {
                                                                    x: cx,
                                                                    y: cy - 10,
                                                                    textAnchor: "middle",
                                                                    fill: "white",
                                                                    fontSize: "8px",
                                                                    fontWeight: "bold",
                                                                    children: label
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                    lineNumber: 744,
                                                                    columnNumber: 29
                                                                }, this)
                                                            ]
                                                        }, `backline-${idx}`, true, {
                                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                            lineNumber: 711,
                                                            columnNumber: 27
                                                        }, this));
                                                    });
                                                    return renders;
                                                })(),
                                                filteredPlayers.map((p, pIdx)=>{
                                                    if (p.x_canvas === undefined || p.y_canvas === undefined) return null;
                                                    const x_screen = p.x_canvas * geometry.scaleX;
                                                    // Calculate feet position: bottom center of bbox, or offset center
                                                    const y_feet = p.bbox ? p.bbox[3] * geometry.scaleY : (p.y_canvas + 15) * geometry.scaleY;
                                                    // Calculate head position: top center of bbox, or offset center
                                                    const y_head = p.bbox ? p.bbox[1] * geometry.scaleY : (p.y_canvas - 15) * geometry.scaleY;
                                                    const color = getTeamColor(p.team_id || p.team);
                                                    const speed = getPlayerSpeed(p, pIdx);
                                                    if (!p.team_id && !p.team) return null;
                                                    const isDefender = backLineDefenders.team0.some((d)=>d.id === p.id) || backLineDefenders.team1.some((d)=>d.id === p.id);
                                                    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("g", {
                                                        children: [
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("ellipse", {
                                                                cx: x_screen,
                                                                cy: y_feet,
                                                                rx: isDefender ? 22 : 18,
                                                                ry: isDefender ? 7.5 : 6,
                                                                stroke: color,
                                                                strokeWidth: 2,
                                                                fill: `${color}22`,
                                                                filter: "drop-shadow(0 0 3px rgba(0,0,0,0.5))"
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                lineNumber: 782,
                                                                columnNumber: 27
                                                            }, this),
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("line", {
                                                                x1: x_screen,
                                                                y1: y_feet,
                                                                x2: x_screen,
                                                                y2: y_head,
                                                                stroke: color,
                                                                strokeWidth: 1,
                                                                strokeDasharray: "2,2",
                                                                opacity: 0.4
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                lineNumber: 794,
                                                                columnNumber: 27
                                                            }, this),
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("g", {
                                                                children: [
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("rect", {
                                                                        x: x_screen - 26,
                                                                        y: y_head - 26,
                                                                        width: 52,
                                                                        height: 11,
                                                                        rx: 2,
                                                                        fill: color,
                                                                        opacity: 0.9
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                        lineNumber: 808,
                                                                        columnNumber: 29
                                                                    }, this),
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("text", {
                                                                        x: x_screen,
                                                                        y: y_head - 18,
                                                                        textAnchor: "middle",
                                                                        fill: "white",
                                                                        fontSize: "8px",
                                                                        fontWeight: "bold",
                                                                        children: useAltNames && dictionary[`P${p.id}`] ? dictionary[`P${p.id}`] : `PLAYER ${p.id}`
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                        lineNumber: 818,
                                                                        columnNumber: 29
                                                                    }, this),
                                                                    speed > 0 && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("text", {
                                                                        x: x_screen,
                                                                        y: y_head - 6,
                                                                        textAnchor: "middle",
                                                                        fill: "#ffffff",
                                                                        fontSize: "9px",
                                                                        fontWeight: "bold",
                                                                        filter: "drop-shadow(0px 1px 2px rgba(0,0,0,0.95))",
                                                                        children: `${speed.toFixed(1)} km/h`
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                        lineNumber: 831,
                                                                        columnNumber: 31
                                                                    }, this)
                                                                ]
                                                            }, void 0, true, {
                                                                fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                lineNumber: 806,
                                                                columnNumber: 27
                                                            }, this)
                                                        ]
                                                    }, `player-ann-${p.id || pIdx}`, true, {
                                                        fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                        lineNumber: 780,
                                                        columnNumber: 25
                                                    }, this);
                                                })
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                            lineNumber: 604,
                                            columnNumber: 19
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "pointer-events-none absolute inset-0 flex items-center justify-center bg-black/20",
                                            children: !isPlaying && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: "flex h-16 w-16 items-center justify-center rounded-full bg-white/20 backdrop-blur-sm transition-transform hover:scale-110",
                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                                                    className: "ml-1 h-8 w-8 text-white",
                                                    fill: "currentColor",
                                                    viewBox: "0 0 24 24",
                                                    children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                                        d: "M8 5v14l11-7z"
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                        lineNumber: 854,
                                                        columnNumber: 25
                                                    }, this)
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                    lineNumber: 853,
                                                    columnNumber: 23
                                                }, this)
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                lineNumber: 852,
                                                columnNumber: 21
                                            }, this)
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                            lineNumber: 850,
                                            columnNumber: 17
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                    lineNumber: 581,
                                    columnNumber: 15
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "flex shrink-0 items-center gap-2 border-t border-gray-800 bg-black/90 px-2 py-2",
                                    onClick: (e)=>e.stopPropagation(),
                                    onKeyDown: (e)=>e.stopPropagation(),
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                            type: "button",
                                            onClick: (e)=>{
                                                e.stopPropagation();
                                                togglePlay();
                                            },
                                            className: "flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-white/10 text-white transition hover:bg-white/20",
                                            "aria-label": isPlaying ? "Pause" : "Play",
                                            children: isPlaying ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                                                className: "h-5 w-5",
                                                fill: "currentColor",
                                                viewBox: "0 0 24 24",
                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                                    d: "M6 19h4V5H6v14zm8-14v14h4V5h-4z"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                    lineNumber: 877,
                                                    columnNumber: 23
                                                }, this)
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                lineNumber: 876,
                                                columnNumber: 21
                                            }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                                                className: "ml-0.5 h-5 w-5",
                                                fill: "currentColor",
                                                viewBox: "0 0 24 24",
                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                                    d: "M8 5v14l11-7z"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                    lineNumber: 881,
                                                    columnNumber: 23
                                                }, this)
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                lineNumber: 880,
                                                columnNumber: 21
                                            }, this)
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                            lineNumber: 866,
                                            columnNumber: 17
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                            type: "button",
                                            onClick: (e)=>{
                                                e.stopPropagation();
                                                setShowOverlays((prev)=>!prev);
                                            },
                                            className: `flex h-9 px-2.5 shrink-0 items-center justify-center gap-1.5 rounded-md text-xs font-bold transition ${showOverlays ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/30" : "bg-white/10 text-gray-400 hover:bg-white/20"}`,
                                            "aria-label": "Toggle Tactical Overlays",
                                            children: [
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                                                    className: "h-4 w-4",
                                                    fill: "none",
                                                    stroke: "currentColor",
                                                    strokeWidth: "2",
                                                    viewBox: "0 0 24 24",
                                                    children: [
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                                            strokeLinecap: "round",
                                                            strokeLinejoin: "round",
                                                            d: "M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                            lineNumber: 899,
                                                            columnNumber: 21
                                                        }, this),
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                                            strokeLinecap: "round",
                                                            strokeLinejoin: "round",
                                                            d: "M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                            lineNumber: 900,
                                                            columnNumber: 21
                                                        }, this)
                                                    ]
                                                }, void 0, true, {
                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                    lineNumber: 898,
                                                    columnNumber: 19
                                                }, this),
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                    children: "Overlays"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                    lineNumber: 902,
                                                    columnNumber: 19
                                                }, this)
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                            lineNumber: 885,
                                            columnNumber: 17
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                            type: "button",
                                            onClick: (e)=>{
                                                e.stopPropagation();
                                                toggleFullscreen();
                                            },
                                            className: "flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-white/10 text-white transition hover:bg-white/20",
                                            "aria-label": isFullscreen ? "Exit Fullscreen" : "Fullscreen",
                                            children: isFullscreen ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                                                className: "h-5 w-5",
                                                fill: "none",
                                                stroke: "currentColor",
                                                strokeWidth: "2",
                                                viewBox: "0 0 24 24",
                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                                    strokeLinecap: "round",
                                                    strokeLinejoin: "round",
                                                    d: "M9 9L3 3m0 0l2 2m-2-2h5M15 9l6-6m0 0l-2 2m2-2h-5M9 15l-6 6m0 0l2-2m-2 2h5M15 15l6 6m0 0l-2-2m2-2h-5"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                    lineNumber: 915,
                                                    columnNumber: 23
                                                }, this)
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                lineNumber: 914,
                                                columnNumber: 21
                                            }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                                                className: "h-5 w-5",
                                                fill: "none",
                                                stroke: "currentColor",
                                                strokeWidth: "2",
                                                viewBox: "0 0 24 24",
                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                                    strokeLinecap: "round",
                                                    strokeLinejoin: "round",
                                                    d: "M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5m-7 11H4m0 0v-4m0 4l5-5m11 5v-4m0 4h-4m4 0l-5-5"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                    lineNumber: 919,
                                                    columnNumber: 23
                                                }, this)
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                lineNumber: 918,
                                                columnNumber: 21
                                            }, this)
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                            lineNumber: 904,
                                            columnNumber: 17
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                            type: "range",
                                            min: 0,
                                            max: maxT > 0 ? maxT : 0,
                                            step: "any",
                                            value: maxT > 0 ? rangeValue : 0,
                                            onChange: onSeekInput,
                                            className: "h-2 min-w-0 flex-1 cursor-pointer accent-emerald-500",
                                            "aria-label": "Seek video",
                                            disabled: maxT <= 0
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                            lineNumber: 923,
                                            columnNumber: 17
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                            className: "shrink-0 font-mono text-xs tabular-nums text-gray-300",
                                            children: [
                                                formatClock(currentTime),
                                                " / ",
                                                formatClock(duration)
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                            lineNumber: 934,
                                            columnNumber: 17
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                    lineNumber: 861,
                                    columnNumber: 15
                                }, this),
                                videoSrc && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "border-t border-gray-900 bg-[#080d08]/85 px-4 py-3",
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h4", {
                                            className: "text-[9px] font-bold font-mono uppercase tracking-[0.2em] text-gray-500 mb-2",
                                            children: "Tactical Action Clips"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                            lineNumber: 940,
                                            columnNumber: 19
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "flex gap-3 overflow-x-auto pb-1 [&::-webkit-scrollbar]:h-1 [&::-webkit-scrollbar-thumb]:bg-gray-800 scrollbar-thin scrollbar-thumb-gray-800",
                                            children: detectedClips.map((clip)=>{
                                                return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                    type: "button",
                                                    onClick: ()=>playClip(clip.time),
                                                    className: "flex min-w-[210px] items-center gap-3 rounded-xl border border-gray-850 bg-[#060a06]/90 p-2.5 text-left transition hover:scale-[1.02] hover:border-emerald-500/40 hover:bg-[#111a12]/50 active:scale-[0.98] group",
                                                    children: [
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                            className: "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-400 group-hover:bg-emerald-500/20 transition-colors",
                                                            children: clip.type === "pass" ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                                                                className: "h-4 w-4",
                                                                fill: "none",
                                                                stroke: "currentColor",
                                                                strokeWidth: "2",
                                                                viewBox: "0 0 24 24",
                                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                                                    strokeLinecap: "round",
                                                                    strokeLinejoin: "round",
                                                                    d: "M13 5l7 7-7 7M5 5l7 7-7 7"
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                    lineNumber: 955,
                                                                    columnNumber: 33
                                                                }, this)
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                lineNumber: 954,
                                                                columnNumber: 31
                                                            }, this) : clip.type === "triangle" ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                                                                className: "h-4 w-4",
                                                                fill: "none",
                                                                stroke: "currentColor",
                                                                strokeWidth: "2",
                                                                viewBox: "0 0 24 24",
                                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                                                    strokeLinecap: "round",
                                                                    strokeLinejoin: "round",
                                                                    d: "M4 19L12 5l8 14H4z"
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                    lineNumber: 959,
                                                                    columnNumber: 33
                                                                }, this)
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                lineNumber: 958,
                                                                columnNumber: 31
                                                            }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                                                                className: "h-4 w-4",
                                                                fill: "none",
                                                                stroke: "currentColor",
                                                                strokeWidth: "2",
                                                                viewBox: "0 0 24 24",
                                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                                                    strokeLinecap: "round",
                                                                    strokeLinejoin: "round",
                                                                    d: "M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                    lineNumber: 963,
                                                                    columnNumber: 33
                                                                }, this)
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                lineNumber: 962,
                                                                columnNumber: 31
                                                            }, this)
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                            lineNumber: 952,
                                                            columnNumber: 27
                                                        }, this),
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                            className: "min-w-0 flex-1",
                                                            children: [
                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                    className: "flex items-center justify-between gap-1",
                                                                    children: [
                                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                            className: "truncate text-xs font-bold text-white group-hover:text-emerald-400 transition-colors",
                                                                            children: clip.title
                                                                        }, void 0, false, {
                                                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                            lineNumber: 969,
                                                                            columnNumber: 31
                                                                        }, this),
                                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                            className: "shrink-0 font-mono text-[9px] font-bold text-gray-500",
                                                                            children: formatClock(clip.time)
                                                                        }, void 0, false, {
                                                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                            lineNumber: 972,
                                                                            columnNumber: 31
                                                                        }, this)
                                                                    ]
                                                                }, void 0, true, {
                                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                    lineNumber: 968,
                                                                    columnNumber: 29
                                                                }, this),
                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                    className: "block truncate text-[10px] text-gray-500",
                                                                    children: clip.desc
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                                    lineNumber: 976,
                                                                    columnNumber: 29
                                                                }, this)
                                                            ]
                                                        }, void 0, true, {
                                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                            lineNumber: 967,
                                                            columnNumber: 27
                                                        }, this)
                                                    ]
                                                }, clip.id, true, {
                                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                                    lineNumber: 946,
                                                    columnNumber: 25
                                                }, this);
                                            })
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                            lineNumber: 943,
                                            columnNumber: 19
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                    lineNumber: 939,
                                    columnNumber: 17
                                }, this)
                            ]
                        }, void 0, true) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "flex flex-1 flex-col items-center justify-center gap-4 p-6 text-center text-gray-500",
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "rounded-full bg-gray-900 p-6",
                                    children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("svg", {
                                        className: "h-12 w-12 opacity-50",
                                        fill: "none",
                                        stroke: "currentColor",
                                        viewBox: "0 0 24 24",
                                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("path", {
                                            strokeLinecap: "round",
                                            strokeLinejoin: "round",
                                            strokeWidth: 1,
                                            d: "M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                            lineNumber: 991,
                                            columnNumber: 19
                                        }, this)
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                        lineNumber: 990,
                                        columnNumber: 17
                                    }, this)
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                    lineNumber: 989,
                                    columnNumber: 15
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                            className: "text-sm font-medium text-gray-400",
                                            children: "No video loaded"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                            lineNumber: 1000,
                                            columnNumber: 17
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                            className: "mt-1 text-xs",
                                            children: "Upload a clip to begin analysis"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                            lineNumber: 1001,
                                            columnNumber: 17
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                    lineNumber: 999,
                                    columnNumber: 15
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                            lineNumber: 988,
                            columnNumber: 13
                        }, this)
                    ]
                }, void 0, true, {
                    fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                    lineNumber: 567,
                    columnNumber: 9
                }, this)
            }, void 0, false, {
                fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                lineNumber: 561,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "flex items-center justify-between px-1",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                        className: "text-[10px] font-bold uppercase tracking-wider text-gray-500",
                        children: "Source"
                    }, void 0, false, {
                        fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                        lineNumber: 1009,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex items-center gap-2",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                className: `text-[10px] font-bold uppercase ${status === "completed" ? "text-emerald-500" : "text-gray-600"}`,
                                children: status
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                lineNumber: 1011,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                type: "button",
                                onClick: onDownload,
                                disabled: status !== "completed",
                                className: "rounded border border-gray-800 bg-gray-900 px-2 py-1 text-[10px] font-bold uppercase text-gray-300 transition hover:bg-gray-800 disabled:opacity-50",
                                children: "Download JSON"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                                lineNumber: 1018,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                        lineNumber: 1010,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/VideoHUD.tsx",
                lineNumber: 1008,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/app/workspace/components/VideoHUD.tsx",
        lineNumber: 560,
        columnNumber: 5
    }, this);
}
_s(VideoHUD, "ex8SBZtMKVYAVX+zTCC/j9UICOM=");
_c = VideoHUD;
var _c;
__turbopack_context__.k.register(_c, "VideoHUD");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/hooks/useVideoFrameSync.ts [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "useVideoFrameSync",
    ()=>useVideoFrameSync
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var _s = __turbopack_context__.k.signature();
;
const DEFAULT_FPS = 30;
const MAX_REF_RETRY_FRAMES = 360;
function useVideoFrameSync({ videoRef, fps, maxFrameIndex, onFrameChange }) {
    _s();
    const rafIdRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    const lastFrameRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(-1);
    const fpsRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(fps);
    const maxFrameRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(maxFrameIndex);
    const onFrameRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(onFrameChange);
    fpsRef.current = fps;
    maxFrameRef.current = maxFrameIndex;
    onFrameRef.current = onFrameChange;
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "useVideoFrameSync.useEffect": ()=>{
            let cancelled = false;
            let retryRafId = null;
            let retryCount = 0;
            let detachVideo = null;
            const stop = {
                "useVideoFrameSync.useEffect.stop": ()=>{
                    if (rafIdRef.current != null) {
                        cancelAnimationFrame(rafIdRef.current);
                        rafIdRef.current = null;
                    }
                }
            }["useVideoFrameSync.useEffect.stop"];
            const safeFps = {
                "useVideoFrameSync.useEffect.safeFps": ()=>{
                    const f = fpsRef.current;
                    if (Number.isFinite(f) && f > 0) return f;
                    return DEFAULT_FPS;
                }
            }["useVideoFrameSync.useEffect.safeFps"];
            const emitCurrent = {
                "useVideoFrameSync.useEffect.emitCurrent": (video)=>{
                    const curFps = safeFps();
                    const t = video.currentTime;
                    const timeOk = Number.isFinite(t) && t >= 0 ? t : 0;
                    const rawIdx = Math.round(timeOk * curFps);
                    const idx = Math.max(0, Math.min(maxFrameRef.current, Number.isFinite(rawIdx) ? rawIdx : 0));
                    if (idx !== lastFrameRef.current) {
                        lastFrameRef.current = idx;
                        onFrameRef.current(idx);
                    }
                }
            }["useVideoFrameSync.useEffect.emitCurrent"];
            const tick = {
                "useVideoFrameSync.useEffect.tick": (video)=>{
                    emitCurrent(video);
                    if (!cancelled && !video.paused && !video.ended) {
                        rafIdRef.current = requestAnimationFrame({
                            "useVideoFrameSync.useEffect.tick": ()=>tick(video)
                        }["useVideoFrameSync.useEffect.tick"]);
                    } else {
                        rafIdRef.current = null;
                    }
                }
            }["useVideoFrameSync.useEffect.tick"];
            const attach = {
                "useVideoFrameSync.useEffect.attach": (video)=>{
                    const onPlay = {
                        "useVideoFrameSync.useEffect.attach.onPlay": ()=>{
                            stop();
                            rafIdRef.current = requestAnimationFrame({
                                "useVideoFrameSync.useEffect.attach.onPlay": ()=>tick(video)
                            }["useVideoFrameSync.useEffect.attach.onPlay"]);
                        }
                    }["useVideoFrameSync.useEffect.attach.onPlay"];
                    const onPauseOrEnd = {
                        "useVideoFrameSync.useEffect.attach.onPauseOrEnd": ()=>stop()
                    }["useVideoFrameSync.useEffect.attach.onPauseOrEnd"];
                    const onSeeked = {
                        "useVideoFrameSync.useEffect.attach.onSeeked": ()=>emitCurrent(video)
                    }["useVideoFrameSync.useEffect.attach.onSeeked"];
                    const onSeeking = {
                        "useVideoFrameSync.useEffect.attach.onSeeking": ()=>emitCurrent(video)
                    }["useVideoFrameSync.useEffect.attach.onSeeking"];
                    const onTimeUpdate = {
                        "useVideoFrameSync.useEffect.attach.onTimeUpdate": ()=>emitCurrent(video)
                    }["useVideoFrameSync.useEffect.attach.onTimeUpdate"];
                    video.addEventListener("play", onPlay);
                    video.addEventListener("pause", onPauseOrEnd);
                    video.addEventListener("ended", onPauseOrEnd);
                    video.addEventListener("seeked", onSeeked);
                    video.addEventListener("seeking", onSeeking);
                    video.addEventListener("timeupdate", onTimeUpdate);
                    emitCurrent(video);
                    if (!video.paused && !video.ended) {
                        rafIdRef.current = requestAnimationFrame({
                            "useVideoFrameSync.useEffect.attach": ()=>tick(video)
                        }["useVideoFrameSync.useEffect.attach"]);
                    }
                    return ({
                        "useVideoFrameSync.useEffect.attach": ()=>{
                            stop();
                            video.removeEventListener("play", onPlay);
                            video.removeEventListener("pause", onPauseOrEnd);
                            video.removeEventListener("ended", onPauseOrEnd);
                            video.removeEventListener("seeked", onSeeked);
                            video.removeEventListener("seeking", onSeeking);
                            video.removeEventListener("timeupdate", onTimeUpdate);
                        }
                    })["useVideoFrameSync.useEffect.attach"];
                }
            }["useVideoFrameSync.useEffect.attach"];
            const tryAttach = {
                "useVideoFrameSync.useEffect.tryAttach": ()=>{
                    if (cancelled) return;
                    const video = videoRef.current;
                    if (!video) {
                        if (retryCount < MAX_REF_RETRY_FRAMES) {
                            retryCount += 1;
                            retryRafId = requestAnimationFrame(tryAttach);
                        }
                        return;
                    }
                    retryRafId = null;
                    detachVideo = attach(video);
                }
            }["useVideoFrameSync.useEffect.tryAttach"];
            tryAttach();
            return ({
                "useVideoFrameSync.useEffect": ()=>{
                    cancelled = true;
                    if (retryRafId != null) {
                        cancelAnimationFrame(retryRafId);
                        retryRafId = null;
                    }
                    stop();
                    detachVideo?.();
                    detachVideo = null;
                }
            })["useVideoFrameSync.useEffect"];
        }
    }["useVideoFrameSync.useEffect"], [
        videoRef,
        fps,
        maxFrameIndex
    ]);
}
_s(useVideoFrameSync, "qShD6VIucAzB/xmWj3jtWGpnPyY=");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/lib/trackingAdapter.ts [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "PITCH_LENGTH_M",
    ()=>PITCH_LENGTH_M,
    "PITCH_WIDTH_M",
    ()=>PITCH_WIDTH_M,
    "adaptTrackingPayload",
    ()=>adaptTrackingPayload,
    "buildFrameLookup",
    ()=>buildFrameLookup,
    "normalizePitchMeters",
    ()=>normalizePitchMeters,
    "projectMetersToCanvas",
    ()=>projectMetersToCanvas
]);
const PITCH_LENGTH_M = 105;
const PITCH_WIDTH_M = 68;
const LEGACY_SCALE = 10;
function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}
function normalizePitchMeters(xPitch, yPitch) {
    if (xPitch == null || yPitch == null) return null;
    let xM = xPitch;
    let yM = yPitch;
    // If coordinates look like they are in legacy 10x units (e.g. 525 instead of 52.5)
    if (Math.abs(xM) > PITCH_LENGTH_M + 5 || Math.abs(yM) > PITCH_WIDTH_M + 5) {
        xM = xM / LEGACY_SCALE;
        yM = yM / LEGACY_SCALE;
    }
    // No longer shifting centered coordinates as it caused regressions
    // with standard 0-105m tracking data.
    return {
        xM: clamp(xM, 0, PITCH_LENGTH_M),
        yM: clamp(yM, 0, PITCH_WIDTH_M)
    };
}
function projectMetersToCanvas(xM, yM, widthPx, heightPx, paddingPx) {
    const innerW = Math.max(1, widthPx - paddingPx * 2);
    const innerH = Math.max(1, heightPx - paddingPx * 2);
    const xPx = paddingPx + clamp(xM, 0, PITCH_LENGTH_M) / PITCH_LENGTH_M * innerW;
    const yPx = paddingPx + clamp(yM, 0, PITCH_WIDTH_M) / PITCH_WIDTH_M * innerH;
    return {
        xPx,
        yPx
    };
}
function buildFrameLookup(payload) {
    const frames = payload.frames ?? [];
    const frameByIdx = new Map();
    for (const frame of frames)frameByIdx.set(frame.frame_idx, frame);
    const totalFrames = frames.length;
    return {
        totalFrames,
        getFrameByIndex: (frameIndex)=>frameByIdx.get(frameIndex) ?? frames[frameIndex]
    };
}
function adaptTrackingPayload(payload) {
    const t = payload.telemetry;
    if (t?.player_position_space !== "radar_pixels") {
        return payload;
    }
    const ppm = t.radar_pixels_per_meter ?? 10;
    if (!Number.isFinite(ppm) || ppm <= 0) {
        return payload;
    }
    const frames = (payload.frames ?? []).map((fr)=>({
            ...fr,
            players: fr.players.map((p)=>({
                    ...p,
                    x_pitch: p.x_pitch != null && p.y_pitch != null ? p.x_pitch / ppm : p.x_pitch,
                    y_pitch: p.x_pitch != null && p.y_pitch != null ? p.y_pitch / ppm : p.y_pitch
                })),
            ball_xy: fr.ball_xy != null ? [
                fr.ball_xy[0] / ppm,
                fr.ball_xy[1] / ppm
            ] : null
        }));
    return {
        ...payload,
        telemetry: {
            ...t,
            player_position_space: "meters"
        },
        frames
    };
}
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/app/workspace/components/radar/pitch.ts [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "drawPitchBackground",
    ()=>drawPitchBackground
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/trackingAdapter.ts [app-client] (ecmascript)");
;
const PITCH_GREEN_TOP = "#0d2b15";
const PITCH_GREEN_BOTTOM = "#061a0d";
const PITCH_LINE = "rgba(255, 255, 255, 0.8)";
const PITCH_LINE_GLOW = "rgba(16, 185, 129, 0.2)";
const LINE_WIDTH = 1.5;
function strokeRectMeters(ctx, xM, yM, wM, hM, widthPx, heightPx, paddingPx) {
    const topLeft = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["projectMetersToCanvas"])(xM, yM, widthPx, heightPx, paddingPx);
    const bottomRight = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["projectMetersToCanvas"])(xM + wM, yM + hM, widthPx, heightPx, paddingPx);
    ctx.strokeRect(topLeft.xPx, topLeft.yPx, bottomRight.xPx - topLeft.xPx, bottomRight.yPx - topLeft.yPx);
}
function drawPitchBackground(ctx, widthPx, heightPx, paddingPx) {
    // 1. Clear and create premium gradient background
    ctx.clearRect(0, 0, widthPx, heightPx);
    const gradient = ctx.createRadialGradient(widthPx / 2, heightPx / 2, 0, widthPx / 2, heightPx / 2, widthPx / 1.5);
    gradient.addColorStop(0, PITCH_GREEN_TOP);
    gradient.addColorStop(1, PITCH_GREEN_BOTTOM);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, widthPx, heightPx);
    // 2. Add subtle grid pattern
    ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
    ctx.lineWidth = 0.5;
    for(let i = 0; i < 10; i++){
        const x = paddingPx + i * (widthPx - paddingPx * 2) / 9;
        ctx.beginPath();
        ctx.moveTo(x, paddingPx);
        ctx.lineTo(x, heightPx - paddingPx);
        ctx.stroke();
    }
    // 3. Set line styles
    ctx.strokeStyle = PITCH_LINE;
    ctx.lineWidth = LINE_WIDTH;
    ctx.shadowBlur = 4;
    ctx.shadowColor = PITCH_LINE_GLOW;
    // Outer boundary.
    strokeRectMeters(ctx, 0, 0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["PITCH_LENGTH_M"], __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["PITCH_WIDTH_M"], widthPx, heightPx, paddingPx);
    // Halfway line.
    const centerTop = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["projectMetersToCanvas"])(__TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["PITCH_LENGTH_M"] / 2, 0, widthPx, heightPx, paddingPx);
    const centerBottom = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["projectMetersToCanvas"])(__TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["PITCH_LENGTH_M"] / 2, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["PITCH_WIDTH_M"], widthPx, heightPx, paddingPx);
    ctx.beginPath();
    ctx.moveTo(centerTop.xPx, centerTop.yPx);
    ctx.lineTo(centerBottom.xPx, centerBottom.yPx);
    ctx.stroke();
    // Center circle and spot.
    const center = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["projectMetersToCanvas"])(__TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["PITCH_LENGTH_M"] / 2, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["PITCH_WIDTH_M"] / 2, widthPx, heightPx, paddingPx);
    const edgePoint = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["projectMetersToCanvas"])(__TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["PITCH_LENGTH_M"] / 2 + 9.15, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["PITCH_WIDTH_M"] / 2, widthPx, heightPx, paddingPx);
    const circleR = Math.max(1, edgePoint.xPx - center.xPx);
    ctx.beginPath();
    ctx.arc(center.xPx, center.yPx, circleR, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.fillStyle = PITCH_LINE;
    ctx.arc(center.xPx, center.yPx, 1.5, 0, Math.PI * 2);
    ctx.fill();
    // Penalty areas.
    const penAreaDepth = 16.5;
    const penAreaWidth = 40.3;
    const penY = (__TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["PITCH_WIDTH_M"] - penAreaWidth) / 2;
    strokeRectMeters(ctx, 0, penY, penAreaDepth, penAreaWidth, widthPx, heightPx, paddingPx);
    strokeRectMeters(ctx, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["PITCH_LENGTH_M"] - penAreaDepth, penY, penAreaDepth, penAreaWidth, widthPx, heightPx, paddingPx);
    // Six-yard boxes.
    const sixDepth = 5.5;
    const sixWidth = 18.32;
    const sixY = (__TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["PITCH_WIDTH_M"] - sixWidth) / 2;
    strokeRectMeters(ctx, 0, sixY, sixDepth, sixWidth, widthPx, heightPx, paddingPx);
    strokeRectMeters(ctx, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["PITCH_LENGTH_M"] - sixDepth, sixY, sixDepth, sixWidth, widthPx, heightPx, paddingPx);
    // Reset shadow for players
    ctx.shadowBlur = 0;
}
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/app/workspace/components/radar/RadarCanvas.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>RadarCanvas
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/trackingAdapter.ts [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$radar$2f$pitch$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/app/workspace/components/radar/pitch.ts [app-client] (ecmascript)");
;
var _s = __turbopack_context__.k.signature();
"use client";
;
;
;
const TEAM_0_COLOR = "#3b82f6"; // Blue
const TEAM_1_COLOR = "#ef4444"; // Red
const UNKNOWN_COLOR = "#6b7280";
const BALL_COLOR = "#ffffff";
function RadarCanvas({ frame }) {
    _s();
    const canvasRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    const containerRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    const [dimensions, setDimensions] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])({
        width: 800,
        height: 518
    });
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "RadarCanvas.useEffect": ()=>{
            const updateSize = {
                "RadarCanvas.useEffect.updateSize": ()=>{
                    if (!containerRef.current) return;
                    const { width } = containerRef.current.getBoundingClientRect();
                    // Restore standard pitch aspect ratio 105:68
                    const height = Math.floor(width * 68 / 105);
                    setDimensions({
                        width,
                        height
                    });
                }
            }["RadarCanvas.useEffect.updateSize"];
            updateSize();
            window.addEventListener("resize", updateSize);
            return ({
                "RadarCanvas.useEffect": ()=>window.removeEventListener("resize", updateSize)
            })["RadarCanvas.useEffect"];
        }
    }["RadarCanvas.useEffect"], []);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "RadarCanvas.useEffect": ()=>{
            const canvas = canvasRef.current;
            if (!canvas) return;
            const ctx = canvas.getContext("2d");
            if (!ctx) return;
            const dpr = window.devicePixelRatio || 1;
            canvas.width = dimensions.width * dpr;
            canvas.height = dimensions.height * dpr;
            ctx.scale(dpr, dpr);
            const { width, height } = dimensions;
            const padding = Math.max(12, width * 0.04);
            (0, __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$radar$2f$pitch$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["drawPitchBackground"])(ctx, width, height, padding);
            if (!frame) return;
            // Draw Players
            for (const player of frame.players){
                if (player.id === 75 || player.id === 76 || player.id === 79) continue;
                const normalized = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["normalizePitchMeters"])(player.x_pitch, player.y_pitch);
                if (!normalized) continue;
                const { xPx, yPx } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["projectMetersToCanvas"])(normalized.xM, normalized.yM, width, height, padding);
                const color = player.team_id === "team_0" ? TEAM_0_COLOR : player.team_id === "team_1" ? TEAM_1_COLOR : UNKNOWN_COLOR;
                ctx.beginPath();
                ctx.arc(xPx, yPx, width * 0.008, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.strokeStyle = "white";
                ctx.lineWidth = 1;
                ctx.stroke();
                // Clean ID label
                ctx.fillStyle = "white";
                ctx.font = `bold ${Math.max(8, width * 0.012)}px Inter, sans-serif`;
                ctx.textAlign = "center";
                ctx.fillText(player.id ? player.id.toString() : "", xPx, yPx - width * 0.015);
            }
            // Draw Ball
            if (frame.ball_xy) {
                const [bx, by] = frame.ball_xy;
                const normalized = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["normalizePitchMeters"])(bx, by);
                if (normalized) {
                    const { xPx, yPx } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["projectMetersToCanvas"])(normalized.xM, normalized.yM, width, height, padding);
                    ctx.fillStyle = BALL_COLOR;
                    ctx.beginPath();
                    ctx.arc(xPx, yPx, width * 0.006, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.strokeStyle = "black";
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
        }
    }["RadarCanvas.useEffect"], [
        frame,
        dimensions
    ]);
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        ref: containerRef,
        className: "w-full",
        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("canvas", {
            ref: canvasRef,
            style: {
                width: "100%",
                height: "auto",
                display: "block",
                aspectRatio: "105 / 68"
            },
            className: "rounded-lg border border-gray-800 bg-[#061a0d] shadow-lg"
        }, void 0, false, {
            fileName: "[project]/app/workspace/components/radar/RadarCanvas.tsx",
            lineNumber: 118,
            columnNumber: 7
        }, this)
    }, void 0, false, {
        fileName: "[project]/app/workspace/components/radar/RadarCanvas.tsx",
        lineNumber: 117,
        columnNumber: 5
    }, this);
}
_s(RadarCanvas, "gKubDU3XMxAD1IJCrmoZoJUIWZE=");
_c = RadarCanvas;
var _c;
__turbopack_context__.k.register(_c, "RadarCanvas");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/app/workspace/components/radar/RadarWidget.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>RadarWidget
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useVideoFrameSync$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/hooks/useVideoFrameSync.ts [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/trackingAdapter.ts [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$radar$2f$RadarCanvas$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/app/workspace/components/radar/RadarCanvas.tsx [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/apiBase.ts [app-client] (ecmascript)");
;
var _s = __turbopack_context__.k.signature();
;
;
;
;
;
function RadarWidget({ videoRef, trackingData, onFrameChange }) {
    _s();
    const adaptedTracking = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "RadarWidget.useMemo[adaptedTracking]": ()=>trackingData ? (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["adaptTrackingPayload"])(trackingData) : null
    }["RadarWidget.useMemo[adaptedTracking]"], [
        trackingData
    ]);
    const lookup = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "RadarWidget.useMemo[lookup]": ()=>adaptedTracking ? (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$trackingAdapter$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["buildFrameLookup"])(adaptedTracking) : null
    }["RadarWidget.useMemo[lookup]"], [
        adaptedTracking
    ]);
    const maxFrameIndex = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "RadarWidget.useMemo[maxFrameIndex]": ()=>{
            if (!lookup) return 0;
            const n = lookup.totalFrames;
            const tel = adaptedTracking?.telemetry?.total_frames_processed;
            if (typeof tel === "number" && Number.isFinite(tel) && tel > 0) {
                return Math.max(0, Math.min(Math.floor(tel), n) - 1);
            }
            return Math.max(0, n - 1);
        }
    }["RadarWidget.useMemo[maxFrameIndex]"], [
        adaptedTracking,
        lookup
    ]);
    const [currentFrame, setCurrentFrame] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(0);
    const [syncFps, setSyncFps] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(30);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "RadarWidget.useEffect": ()=>{
            setCurrentFrame(0);
        }
    }["RadarWidget.useEffect"], [
        adaptedTracking
    ]);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "RadarWidget.useEffect": ()=>{
            const video = videoRef.current;
            if (!video || !adaptedTracking) return;
            const totalFrames = adaptedTracking.telemetry?.total_frames_processed ?? adaptedTracking.frames.length;
            const onLoadedMetadata = {
                "RadarWidget.useEffect.onLoadedMetadata": ()=>{
                    if (video.duration > 0 && totalFrames > 0) {
                        const derived = totalFrames / video.duration;
                        setSyncFps(Number.isFinite(derived) && derived > 0 ? derived : 30);
                    } else {
                        setSyncFps(30);
                    }
                }
            }["RadarWidget.useEffect.onLoadedMetadata"];
            video.addEventListener("loadedmetadata", onLoadedMetadata);
            // Metadata may already be available (blob URL warmed before Radar mounts).
            if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
                onLoadedMetadata();
            }
            return ({
                "RadarWidget.useEffect": ()=>video.removeEventListener("loadedmetadata", onLoadedMetadata)
            })["RadarWidget.useEffect"];
        }
    }["RadarWidget.useEffect"], [
        videoRef,
        adaptedTracking
    ]);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useVideoFrameSync$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useVideoFrameSync"])({
        videoRef,
        fps: syncFps,
        maxFrameIndex,
        onFrameChange: setCurrentFrame
    });
    const frame = lookup?.getFrameByIndex(currentFrame);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "RadarWidget.useEffect": ()=>{
            onFrameChange?.(currentFrame, frame);
        }
    }["RadarWidget.useEffect"], [
        currentFrame,
        frame,
        onFrameChange
    ]);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "RadarWidget.useEffect": ()=>{
            // #region agent log
            fetch(`${(0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])()}/ingest/b94af6c0-0f3f-4385-ab39-095f9a480704`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Debug-Session-Id": "bb63ae",
                    ...(0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
                },
                body: JSON.stringify({
                    sessionId: "bb63ae",
                    hypothesisId: "H1",
                    location: "RadarWidget.tsx:mount",
                    message: "RadarWidget mounted",
                    data: {
                        hasTrackingPayload: Boolean(trackingData),
                        adaptedOk: Boolean(adaptedTracking),
                        totalFrames: lookup?.totalFrames ?? 0
                    },
                    timestamp: Date.now()
                })
            }).catch({
                "RadarWidget.useEffect": ()=>{}
            }["RadarWidget.useEffect"]);
        // #endregion
        }
    }["RadarWidget.useEffect"], [
        trackingData,
        adaptedTracking,
        lookup?.totalFrames
    ]);
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        "data-testid": "tactical-radar",
        className: "rounded-lg border border-gray-800 bg-[#0a0f0a] p-3 shadow-sm",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                className: "mb-2 text-sm font-semibold text-gray-200",
                children: "Tactical Radar"
            }, void 0, false, {
                fileName: "[project]/app/workspace/components/radar/RadarWidget.tsx",
                lineNumber: 105,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$radar$2f$RadarCanvas$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"], {
                frame: frame
            }, void 0, false, {
                fileName: "[project]/app/workspace/components/radar/RadarWidget.tsx",
                lineNumber: 106,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "mt-2 flex items-center justify-between text-xs text-gray-500",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                        children: [
                            "Frame ",
                            currentFrame
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/radar/RadarWidget.tsx",
                        lineNumber: 108,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                        children: [
                            "Sync FPS ",
                            syncFps.toFixed(2)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/radar/RadarWidget.tsx",
                        lineNumber: 109,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/radar/RadarWidget.tsx",
                lineNumber: 107,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/app/workspace/components/radar/RadarWidget.tsx",
        lineNumber: 104,
        columnNumber: 5
    }, this);
}
_s(RadarWidget, "k3wZvDJqDXFts3Y/pFcGSau0EHk=", false, function() {
    return [
        __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useVideoFrameSync$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useVideoFrameSync"]
    ];
});
_c = RadarWidget;
var _c;
__turbopack_context__.k.register(_c, "RadarWidget");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/hooks/useStreamingText.ts [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "useStreamingText",
    ()=>useStreamingText
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var _s = __turbopack_context__.k.signature();
;
function useStreamingText(fullText, speedMs = 20) {
    _s();
    const [displayedText, setDisplayedText] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])('');
    const [isStreaming, setIsStreaming] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(false);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "useStreamingText.useEffect": ()=>{
            if (!fullText) {
                setDisplayedText('');
                return;
            }
            setIsStreaming(true);
            setDisplayedText('');
            let currentIndex = 0;
            const intervalId = setInterval({
                "useStreamingText.useEffect.intervalId": ()=>{
                    currentIndex++;
                    setDisplayedText(fullText.substring(0, currentIndex));
                    if (currentIndex >= fullText.length) {
                        setIsStreaming(false);
                        clearInterval(intervalId);
                    }
                }
            }["useStreamingText.useEffect.intervalId"], speedMs);
            return ({
                "useStreamingText.useEffect": ()=>clearInterval(intervalId)
            })["useStreamingText.useEffect"];
        }
    }["useStreamingText.useEffect"], [
        fullText,
        speedMs
    ]);
    return {
        displayedText,
        isStreaming
    };
}
_s(useStreamingText, "XndJlgYVf15jCnlnu6WkaPMEJDM=");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/app/workspace/components/InsightCard.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "InsightCard",
    ()=>InsightCard
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useStreamingText$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/hooks/useStreamingText.ts [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$zap$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Zap$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/zap.mjs [app-client] (ecmascript) <export default as Zap>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$trending$2d$up$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__TrendingUp$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/trending-up.mjs [app-client] (ecmascript) <export default as TrendingUp>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$shield$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Shield$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/shield.mjs [app-client] (ecmascript) <export default as Shield>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$activity$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Activity$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/activity.mjs [app-client] (ecmascript) <export default as Activity>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$film$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Film$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/film.mjs [app-client] (ecmascript) <export default as Film>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$play$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Play$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/play.mjs [app-client] (ecmascript) <export default as Play>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$user$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__User$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/user.mjs [app-client] (ecmascript) <export default as User>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$clock$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Clock$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/clock.mjs [app-client] (ecmascript) <export default as Clock>");
;
var _s = __turbopack_context__.k.signature();
"use client";
;
;
;
const EVENT_NAME_LOOKUP = {
    // Movement
    "MOV_001": "High-Speed Run",
    "MOV_002": "Sprint",
    "MOV_003": "Recovery Run",
    "MOV_004": "Overlap Run",
    "MOV_005": "Underlap Run",
    "MOV_006": "Third-Man Run",
    "MOV_007": "Diagonal Run",
    // Positional
    "POS_001": "Wide Positioning",
    "POS_002": "Half-Space Occupation",
    "POS_003": "Between-Lines",
    "POS_004": "Deep Positioning",
    "POS_005": "Advanced Positioning",
    "POS_006": "Pressing Trap Position",
    // Threat
    "THR_001": "Dangerous Run",
    "THR_002": "Final-Third Entry",
    "THR_003": "Box Entry",
    "THR_004": "Transition Involvement",
    "THR_005": "Dangerous Reception",
    "THR_006": "Channel Exploitation",
    "THR_007": "Isolated Defender Exploit",
    // Shape
    "SHP_001": "High Press Moment",
    "SHP_002": "Mid Block",
    "SHP_003": "Low Block",
    "SHP_004": "Compact Shape",
    "SHP_005": "Stretched Shape",
    "SHP_006": "Overload Zone",
    "SHP_007": "Pressing Trap Triggered",
    "SHP_008": "Counter-Attack Launch",
    // Transition
    "TRN_001": "Defensive Transition",
    "TRN_002": "Offensive Transition",
    "TRN_003": "Press Success",
    "TRN_004": "Press Failure",
    "TRN_005": "Counter-Attack Sequence"
};
function InsightCard({ title, minute, blueTeam, redTeam, metrics, evidenceClips = [], threatContext, eventCountSummary, onPlayClip, useAltNames = false, dictionary = {} }) {
    _s();
    const [activeTeam, setActiveTeam] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])('blue');
    const currentData = activeTeam === 'blue' ? blueTeam : redTeam;
    const { displayedText } = (0, __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useStreamingText$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useStreamingText"])(currentData.payload, 20);
    // blue = team_1, red = team_0  (matches backend naming)
    const teamMetrics = activeTeam === 'blue' ? metrics?.team_1 : metrics?.team_0;
    const compactnessLabel = teamMetrics?.compactness != null ? teamMetrics.compactness > 70 ? "High" : teamMetrics.compactness > 40 ? "Med" : "Low" : "—";
    const transitionLabel = teamMetrics?.transition_speed != null ? teamMetrics.transition_speed > 70 ? "Fast" : teamMetrics.transition_speed > 40 ? "Norm" : "Slow" : "—";
    const tacticalPower = teamMetrics?.tactical_power?.toFixed(1) ?? (activeTeam === 'blue' ? metrics?.team_blue_score?.toFixed(1) : metrics?.team_red_score?.toFixed(1)) ?? "—";
    const winProb = metrics?.win_probability ? activeTeam === 'blue' ? metrics.win_probability.team_blue : metrics.win_probability.team_red : teamMetrics?.win_prob ?? "—";
    const hasTelemetry = evidenceClips && evidenceClips.length > 0 || threatContext?.top_threats && threatContext.top_threats.length > 0 || eventCountSummary && Object.keys(eventCountSummary).length > 0;
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "w-full flex flex-col gap-6 font-sans",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "grid grid-cols-4 gap-4",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "bg-black/30 border border-gray-800 p-3 rounded-lg flex flex-col gap-1",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                className: "text-[9px] font-mono uppercase text-gray-500 tracking-widest",
                                children: "Tactical Power"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                lineNumber: 110,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex items-center justify-between",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "text-sm font-bold text-emerald-400",
                                        children: tacticalPower
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 112,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$activity$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Activity$3e$__["Activity"], {
                                        size: 12,
                                        className: "text-emerald-500/50"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 115,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                lineNumber: 111,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                        lineNumber: 109,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "bg-black/30 border border-gray-800 p-3 rounded-lg flex flex-col gap-1",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                className: "text-[9px] font-mono uppercase text-gray-500 tracking-widest",
                                children: "Win Prob"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                lineNumber: 119,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex items-center justify-between",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "text-sm font-bold text-cyan-400",
                                        children: [
                                            winProb,
                                            winProb !== "—" ? "%" : ""
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 121,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$trending$2d$up$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__TrendingUp$3e$__["TrendingUp"], {
                                        size: 12,
                                        className: "text-cyan-500/50"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 124,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                lineNumber: 120,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                        lineNumber: 118,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "bg-black/30 border border-gray-800 p-3 rounded-lg flex flex-col gap-1",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                className: "text-[9px] font-mono uppercase text-gray-500 tracking-widest",
                                children: "Compactness"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                lineNumber: 128,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex items-center justify-between",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "text-sm font-bold text-amber-400",
                                        children: compactnessLabel
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 130,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$shield$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Shield$3e$__["Shield"], {
                                        size: 12,
                                        className: "text-amber-500/50"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 133,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                lineNumber: 129,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                        lineNumber: 127,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "bg-black/30 border border-gray-800 p-3 rounded-lg flex flex-col gap-1",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                className: "text-[9px] font-mono uppercase text-gray-500 tracking-widest",
                                children: "Trans. Speed"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                lineNumber: 137,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex items-center justify-between",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "text-sm font-bold text-white",
                                        children: transitionLabel
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 139,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$zap$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Zap$3e$__["Zap"], {
                                        size: 12,
                                        className: "text-emerald-500/50"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 142,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                lineNumber: 138,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                        lineNumber: 136,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                lineNumber: 108,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: `grid grid-cols-1 ${hasTelemetry ? 'lg:grid-cols-[1.2fr_0.8fr]' : ''} gap-8`,
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex flex-col gap-4",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex items-center justify-between border-b border-gray-800 pb-2",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h3", {
                                        className: "text-lg font-bold text-gray-100 flex items-center gap-2",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "text-red-500 font-mono",
                                                children: [
                                                    "[",
                                                    minute,
                                                    "']"
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                lineNumber: 154,
                                                columnNumber: 15
                                            }, this),
                                            title
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 153,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "flex bg-gray-900 rounded-md p-0.5 border border-gray-800",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                onClick: ()=>setActiveTeam('blue'),
                                                className: `px-3 py-1 text-[10px] font-bold uppercase rounded ${activeTeam === 'blue' ? 'bg-blue-600/30 text-blue-400' : 'text-gray-500'}`,
                                                children: useAltNames && dictionary["team_1"] ? dictionary["team_1"] : "Blue"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                lineNumber: 158,
                                                columnNumber: 15
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                onClick: ()=>setActiveTeam('red'),
                                                className: `px-3 py-1 text-[10px] font-bold uppercase rounded ${activeTeam === 'red' ? 'bg-red-600/30 text-red-400' : 'text-gray-500'}`,
                                                children: useAltNames && dictionary["team_0"] ? dictionary["team_0"] : "Red"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                lineNumber: 164,
                                                columnNumber: 15
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 157,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                lineNumber: 152,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "bg-[#111a12]/30 border border-emerald-500/10 rounded-xl p-5 relative overflow-hidden flex-1",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "absolute top-0 left-0 w-1 h-full bg-emerald-500/40"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 174,
                                        columnNumber: 14
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "text-gray-300 leading-relaxed text-sm font-medium",
                                        children: [
                                            displayedText,
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "inline-block w-1.5 h-4 bg-emerald-500 ml-1 animate-pulse"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                lineNumber: 177,
                                                columnNumber: 16
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 175,
                                        columnNumber: 14
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                lineNumber: 173,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                        lineNumber: 151,
                        columnNumber: 9
                    }, this),
                    hasTelemetry && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex flex-col gap-4 bg-gray-950/40 border border-gray-900 rounded-2xl p-5 backdrop-blur-sm",
                        children: [
                            evidenceClips && evidenceClips.length > 0 && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "space-y-2.5",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h4", {
                                        className: "text-[10px] font-bold font-mono text-gray-400 uppercase tracking-widest flex items-center gap-2",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$film$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Film$3e$__["Film"], {
                                                size: 12,
                                                className: "text-emerald-500"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                lineNumber: 190,
                                                columnNumber: 19
                                            }, this),
                                            "Visual Evidence Clips"
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 189,
                                        columnNumber: 17
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "space-y-2",
                                        children: evidenceClips.map((clip, idx)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                onClick: ()=>onPlayClip && onPlayClip(clip.start_time_s),
                                                className: "w-full text-left p-3 bg-black/40 hover:bg-[#111a12]/15 border border-gray-900 hover:border-emerald-500/30 rounded-xl transition-all flex items-center justify-between group",
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "flex-1 min-w-0 pr-2",
                                                        children: [
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "text-xs font-semibold text-gray-300 truncate group-hover:text-emerald-400 transition-colors",
                                                                children: clip.label || "Evidence Moment"
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                                lineNumber: 201,
                                                                columnNumber: 25
                                                            }, this),
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "flex items-center gap-2 mt-1",
                                                                children: [
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                        className: "text-[9px] font-mono text-gray-600 flex items-center gap-1",
                                                                        children: [
                                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$clock$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Clock$3e$__["Clock"], {
                                                                                size: 8
                                                                            }, void 0, false, {
                                                                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                                                lineNumber: 206,
                                                                                columnNumber: 29
                                                                            }, this),
                                                                            Math.floor(clip.start_time_s / 60),
                                                                            ":",
                                                                            Math.floor(clip.start_time_s % 60).toString().padStart(2, '0')
                                                                        ]
                                                                    }, void 0, true, {
                                                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                                        lineNumber: 205,
                                                                        columnNumber: 27
                                                                    }, this),
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                        className: "text-[9px] font-mono text-emerald-500/50 bg-emerald-500/5 px-1.5 py-0.5 rounded",
                                                                        children: [
                                                                            Math.round(clip.confidence_pct || clip.relevance_score * 100),
                                                                            "% Match"
                                                                        ]
                                                                    }, void 0, true, {
                                                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                                        lineNumber: 209,
                                                                        columnNumber: 27
                                                                    }, this)
                                                                ]
                                                            }, void 0, true, {
                                                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                                lineNumber: 204,
                                                                columnNumber: 25
                                                            }, this)
                                                        ]
                                                    }, void 0, true, {
                                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                        lineNumber: 200,
                                                        columnNumber: 23
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "h-6 w-6 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-500 group-hover:bg-emerald-500/20 group-hover:scale-105 transition-all",
                                                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$play$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Play$3e$__["Play"], {
                                                            size: 10,
                                                            fill: "currentColor"
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                            lineNumber: 215,
                                                            columnNumber: 25
                                                        }, this)
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                        lineNumber: 214,
                                                        columnNumber: 23
                                                    }, this)
                                                ]
                                            }, idx, true, {
                                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                lineNumber: 195,
                                                columnNumber: 21
                                            }, this))
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 193,
                                        columnNumber: 17
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                lineNumber: 188,
                                columnNumber: 15
                            }, this),
                            threatContext?.top_threats && threatContext.top_threats.length > 0 && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "space-y-2.5 pt-2 border-t border-gray-900",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h4", {
                                        className: "text-[10px] font-bold font-mono text-gray-400 uppercase tracking-widest flex items-center gap-2",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$user$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__User$3e$__["User"], {
                                                size: 12,
                                                className: "text-emerald-500"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                lineNumber: 227,
                                                columnNumber: 19
                                            }, this),
                                            "Key Opponent Threats"
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 226,
                                        columnNumber: 17
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "space-y-2",
                                        children: threatContext.top_threats.map((threat, idx)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: "p-3 bg-black/20 border border-gray-900/80 rounded-xl space-y-1.5",
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "flex items-center justify-between",
                                                        children: [
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "flex items-center gap-2",
                                                                children: [
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                        className: `h-5 w-5 rounded-full flex items-center justify-center text-[10px] font-bold ${threat.team_id === 'team_0' ? 'bg-red-500/20 text-red-400' : 'bg-blue-500/20 text-blue-400'}`,
                                                                        children: [
                                                                            "P",
                                                                            threat.player_id
                                                                        ]
                                                                    }, void 0, true, {
                                                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                                        lineNumber: 235,
                                                                        columnNumber: 27
                                                                    }, this),
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                        className: "text-[10px] font-bold text-gray-400 uppercase tracking-tight",
                                                                        children: useAltNames && dictionary[`P${threat.player_id}`] ? `${dictionary[`P${threat.player_id}`]} (${useAltNames && dictionary[threat.team_id] ? dictionary[threat.team_id] : threat.team_id === 'team_0' ? 'Red' : 'Blue'})` : threat.team_id === 'team_0' ? 'Red Team' : 'Blue Team'
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                                        lineNumber: 238,
                                                                        columnNumber: 27
                                                                    }, this)
                                                                ]
                                                            }, void 0, true, {
                                                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                                lineNumber: 234,
                                                                columnNumber: 25
                                                            }, this),
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                className: "text-[11px] font-mono font-bold text-emerald-400",
                                                                children: [
                                                                    "Threat: ",
                                                                    threat.threat_score.toFixed(1)
                                                                ]
                                                            }, void 0, true, {
                                                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                                lineNumber: 242,
                                                                columnNumber: 25
                                                            }, this)
                                                        ]
                                                    }, void 0, true, {
                                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                        lineNumber: 233,
                                                        columnNumber: 23
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "h-1 bg-gray-900 rounded-full overflow-hidden",
                                                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                            className: `h-full ${threat.team_id === 'team_0' ? 'bg-red-500' : 'bg-blue-500'}`,
                                                            style: {
                                                                width: `${threat.threat_score}%`
                                                            }
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                            lineNumber: 247,
                                                            columnNumber: 25
                                                        }, this)
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                        lineNumber: 246,
                                                        columnNumber: 23
                                                    }, this),
                                                    threat.explanation && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                                        className: "text-[9px] text-gray-500 leading-normal font-sans pt-0.5",
                                                        children: threat.explanation
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                        lineNumber: 253,
                                                        columnNumber: 25
                                                    }, this)
                                                ]
                                            }, idx, true, {
                                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                lineNumber: 232,
                                                columnNumber: 21
                                            }, this))
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 230,
                                        columnNumber: 17
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                lineNumber: 225,
                                columnNumber: 15
                            }, this),
                            eventCountSummary && Object.keys(eventCountSummary).length > 0 && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "space-y-2.5 pt-2 border-t border-gray-900",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h4", {
                                        className: "text-[10px] font-bold font-mono text-gray-400 uppercase tracking-widest flex items-center gap-2",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$activity$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Activity$3e$__["Activity"], {
                                                size: 12,
                                                className: "text-emerald-500"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                lineNumber: 267,
                                                columnNumber: 19
                                            }, this),
                                            "Ontology Triggers"
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 266,
                                        columnNumber: 17
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "flex flex-wrap gap-2",
                                        children: Object.entries(eventCountSummary).map(([code, count])=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: "px-2.5 py-1 bg-black/40 border border-gray-900 hover:border-emerald-500/20 rounded-lg flex items-center gap-2 transition-colors cursor-help",
                                                title: `${code}: ${EVENT_NAME_LOOKUP[code] || 'Unknown Event'}`,
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                        className: "text-[9px] font-mono font-semibold text-gray-400",
                                                        children: (()=>{
                                                            const orig = EVENT_NAME_LOOKUP[code] || code;
                                                            return useAltNames && dictionary[orig] ? dictionary[orig] : orig;
                                                        })()
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                        lineNumber: 277,
                                                        columnNumber: 23
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                        className: "text-[10px] font-mono font-bold text-emerald-400 bg-emerald-500/10 px-1 py-0.25 rounded",
                                                        children: count
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                        lineNumber: 283,
                                                        columnNumber: 23
                                                    }, this)
                                                ]
                                            }, code, true, {
                                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                                lineNumber: 272,
                                                columnNumber: 21
                                            }, this))
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                        lineNumber: 270,
                                        columnNumber: 17
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                                lineNumber: 265,
                                columnNumber: 15
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                        lineNumber: 184,
                        columnNumber: 11
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                lineNumber: 148,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "mt-auto pt-4 border-t border-gray-900 flex justify-between items-center",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                        className: "text-[9px] font-mono text-gray-600 uppercase tracking-widest italic",
                        children: "Tactical Engine Analysis active · v1.2"
                    }, void 0, false, {
                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                        lineNumber: 297,
                        columnNumber: 10
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                        className: "text-[9px] font-mono text-emerald-500/50 uppercase tracking-widest",
                        children: "Gaffer's Guide Standard"
                    }, void 0, false, {
                        fileName: "[project]/app/workspace/components/InsightCard.tsx",
                        lineNumber: 298,
                        columnNumber: 10
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/InsightCard.tsx",
                lineNumber: 296,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/app/workspace/components/InsightCard.tsx",
        lineNumber: 105,
        columnNumber: 5
    }, this);
}
_s(InsightCard, "D3lc74wIjqQUKqnR1YyOd4bnQqI=", false, function() {
    return [
        __TURBOPACK__imported__module__$5b$project$5d2f$hooks$2f$useStreamingText$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useStreamingText"]
    ];
});
_c = InsightCard;
var _c;
__turbopack_context__.k.register(_c, "InsightCard");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/lib/api/reports.ts [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "deleteSavedReport",
    ()=>deleteSavedReport,
    "listSavedReports",
    ()=>listSavedReports,
    "saveTacticalReport",
    ()=>saveTacticalReport
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/apiBase.ts [app-client] (ecmascript)");
;
function getReportsApiBase() {
    return `${(0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])()}/api/v1/elite/reports`;
}
async function saveTacticalReport(jobId, coachAdvice, videoFilename) {
    try {
        // Derive a clean title from the uploaded filename (strip extension)
        const videoTitle = videoFilename ? videoFilename.replace(/\.[^.]+$/, '').replace(/[_-]/g, ' ') : coachAdvice?.video_title || `Match ${new Date().toLocaleDateString()}`;
        const reportData = {
            job_id: jobId,
            timestamp: new Date().toISOString(),
            video_title: videoTitle,
            metadata: coachAdvice?.metadata || {},
            pipeline: coachAdvice?.pipeline || {},
            advice_items: coachAdvice?.advice_items || [],
            summary_data: coachAdvice?.summary_data || {},
            quality_profile: coachAdvice?.quality_profile || 'balanced',
            is_manual_save: true
        };
        const response = await fetch(`${getReportsApiBase()}/save`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...(0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
            },
            body: JSON.stringify(reportData)
        });
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Failed to save report: ${errorText}`);
        }
        return await response.json();
    } catch (error) {
        console.error("Error in saveTacticalReport:", error);
        throw error;
    }
}
async function listSavedReports() {
    try {
        const response = await fetch(`${getReportsApiBase()}`, {
            method: "GET",
            headers: {
                ...(0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
            }
        });
        if (!response.ok) {
            throw new Error("Failed to fetch reports list");
        }
        return await response.json();
    } catch (error) {
        console.error("Error in listSavedReports:", error);
        return [];
    }
}
async function deleteSavedReport(reportId) {
    try {
        const response = await fetch(`${getReportsApiBase()}/${reportId}`, {
            method: "DELETE",
            headers: {
                ...(0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
            }
        });
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Failed to delete report: ${errorText}`);
        }
        return await response.json();
    } catch (error) {
        console.error("Error in deleteSavedReport:", error);
        throw error;
    }
}
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/app/workspace/components/SaveResultsModal.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "SaveResultsModal",
    ()=>SaveResultsModal
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$save$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Save$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/save.mjs [app-client] (ecmascript) <export default as Save>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$download$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Download$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/download.mjs [app-client] (ecmascript) <export default as Download>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$x$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__X$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/x.mjs [app-client] (ecmascript) <export default as X>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$film$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Film$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/film.mjs [app-client] (ecmascript) <export default as Film>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$file$2d$text$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__FileText$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/file-text.mjs [app-client] (ecmascript) <export default as FileText>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$circle$2d$check$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__CheckCircle2$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/circle-check.mjs [app-client] (ecmascript) <export default as CheckCircle2>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2d$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader2$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/loader-circle.mjs [app-client] (ecmascript) <export default as Loader2>");
;
;
function SaveResultsModal({ isOpen, onClose, onSaveBoth, onSaveReportOnly, isSaving, saveStatus }) {
    if (!isOpen) return null;
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm",
        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
            className: "bg-[#0a0f0a] border border-emerald-500/30 rounded-2xl shadow-[0_0_50px_rgba(16,185,129,0.15)] w-full max-w-md overflow-hidden relative animate-in fade-in zoom-in duration-200",
            children: [
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: "p-6 border-b border-gray-900 bg-[#111a12]/50 flex justify-between items-start",
                    children: [
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                                    className: "text-xl font-bold text-white tracking-tight",
                                    children: "Save Tactical Results"
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                    lineNumber: 30,
                                    columnNumber: 13
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                    className: "text-sm text-gray-500 mt-1",
                                    children: "Analysis complete. What would you like to save?"
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                    lineNumber: 31,
                                    columnNumber: 13
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                            lineNumber: 29,
                            columnNumber: 11
                        }, this),
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                            onClick: onClose,
                            className: "text-gray-500 hover:text-white p-1 rounded-lg hover:bg-white/10 transition-colors",
                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$x$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__X$3e$__["X"], {
                                size: 20
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                lineNumber: 37,
                                columnNumber: 13
                            }, this)
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                            lineNumber: 33,
                            columnNumber: 11
                        }, this)
                    ]
                }, void 0, true, {
                    fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                    lineNumber: 28,
                    columnNumber: 9
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: "p-6 space-y-4",
                    children: saveStatus === 'success' ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex flex-col items-center justify-center py-8 text-emerald-500 space-y-4",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$circle$2d$check$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__CheckCircle2$3e$__["CheckCircle2"], {
                                size: 48,
                                className: "animate-in zoom-in duration-300"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                lineNumber: 45,
                                columnNumber: 15
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                className: "font-bold tracking-widest uppercase",
                                children: "Successfully Saved!"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                lineNumber: 46,
                                columnNumber: 15
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                        lineNumber: 44,
                        columnNumber: 13
                    }, this) : saveStatus === 'rendering' ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex flex-col items-center justify-center py-8 text-emerald-500 space-y-4",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2d$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader2$3e$__["Loader2"], {
                                size: 48,
                                className: "animate-spin"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                lineNumber: 50,
                                columnNumber: 15
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                className: "font-bold tracking-widest uppercase text-sm",
                                children: "Rendering Tactical Video..."
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                lineNumber: 51,
                                columnNumber: 15
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                className: "text-xs text-gray-500 text-center max-w-xs",
                                children: "Stitching radar overlay onto your match footage. This may take 30–60 seconds."
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                lineNumber: 52,
                                columnNumber: 15
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                        lineNumber: 49,
                        columnNumber: 13
                    }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                disabled: isSaving,
                                onClick: onSaveBoth,
                                className: "w-full flex items-center gap-4 p-4 rounded-xl border border-emerald-500/50 bg-emerald-500/10 hover:bg-emerald-500/20 transition-all group text-left disabled:opacity-50 disabled:cursor-not-allowed",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "p-3 rounded-full bg-emerald-500 text-black group-hover:scale-110 transition-transform",
                                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$download$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Download$3e$__["Download"], {
                                            size: 24
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                            lineNumber: 63,
                                            columnNumber: 19
                                        }, this)
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                        lineNumber: 62,
                                        columnNumber: 17
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "flex-1",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h3", {
                                                className: "font-bold text-emerald-400",
                                                children: "Save Report & Video"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                                lineNumber: 66,
                                                columnNumber: 19
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                                className: "text-xs text-emerald-500/70 mt-1 flex items-center gap-2",
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$file$2d$text$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__FileText$3e$__["FileText"], {
                                                        size: 12
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                                        lineNumber: 68,
                                                        columnNumber: 21
                                                    }, this),
                                                    " Report ",
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$film$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Film$3e$__["Film"], {
                                                        size: 12,
                                                        className: "ml-2"
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                                        lineNumber: 68,
                                                        columnNumber: 51
                                                    }, this),
                                                    " MP4 Video"
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                                lineNumber: 67,
                                                columnNumber: 19
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                        lineNumber: 65,
                                        columnNumber: 17
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                lineNumber: 57,
                                columnNumber: 15
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                disabled: isSaving,
                                onClick: onSaveReportOnly,
                                className: "w-full flex items-center gap-4 p-4 rounded-xl border border-gray-800 bg-[#111a12] hover:bg-gray-900 hover:border-gray-700 transition-all group text-left disabled:opacity-50 disabled:cursor-not-allowed",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "p-3 rounded-full bg-gray-800 text-gray-400 group-hover:text-white transition-colors",
                                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$save$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Save$3e$__["Save"], {
                                            size: 24
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                            lineNumber: 80,
                                            columnNumber: 19
                                        }, this)
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                        lineNumber: 79,
                                        columnNumber: 17
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "flex-1",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h3", {
                                                className: "font-bold text-gray-300 group-hover:text-white",
                                                children: "Save Report Only"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                                lineNumber: 83,
                                                columnNumber: 19
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                                className: "text-xs text-gray-500 mt-1 flex items-center gap-2",
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$file$2d$text$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__FileText$3e$__["FileText"], {
                                                        size: 12
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                                        lineNumber: 85,
                                                        columnNumber: 21
                                                    }, this),
                                                    " Text data only"
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                                lineNumber: 84,
                                                columnNumber: 19
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                        lineNumber: 82,
                                        columnNumber: 17
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                lineNumber: 74,
                                columnNumber: 15
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                onClick: onClose,
                                disabled: isSaving,
                                className: "w-full py-3 text-sm font-bold text-gray-500 hover:text-white uppercase tracking-widest disabled:opacity-50",
                                children: "Skip for now"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                                lineNumber: 90,
                                columnNumber: 15
                            }, this)
                        ]
                    }, void 0, true)
                }, void 0, false, {
                    fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
                    lineNumber: 42,
                    columnNumber: 9
                }, this)
            ]
        }, void 0, true, {
            fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
            lineNumber: 25,
            columnNumber: 7
        }, this)
    }, void 0, false, {
        fileName: "[project]/app/workspace/components/SaveResultsModal.tsx",
        lineNumber: 24,
        columnNumber: 5
    }, this);
}
_c = SaveResultsModal;
var _c;
__turbopack_context__.k.register(_c, "SaveResultsModal");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/app/workspace/components/TacticalDashboard.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>TacticalDashboard
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$bot$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Bot$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/bot.mjs [app-client] (ecmascript) <export default as Bot>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$user$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__User$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/user.mjs [app-client] (ecmascript) <export default as User>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$send$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Send$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/send.mjs [app-client] (ecmascript) <export default as Send>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$save$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Save$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/save.mjs [app-client] (ecmascript) <export default as Save>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2d$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader2$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/loader-circle.mjs [app-client] (ecmascript) <export default as Loader2>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$activity$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Activity$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/activity.mjs [app-client] (ecmascript) <export default as Activity>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$zap$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Zap$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/zap.mjs [app-client] (ecmascript) <export default as Zap>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$trending$2d$up$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__TrendingUp$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/trending-up.mjs [app-client] (ecmascript) <export default as TrendingUp>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$shield$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Shield$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/shield.mjs [app-client] (ecmascript) <export default as Shield>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$message$2d$square$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__MessageSquare$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/message-square.mjs [app-client] (ecmascript) <export default as MessageSquare>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$clock$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Clock$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/clock.mjs [app-client] (ecmascript) <export default as Clock>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$maximize$2d$2$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Maximize2$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/maximize-2.mjs [app-client] (ecmascript) <export default as Maximize2>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$play$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Play$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/play.mjs [app-client] (ecmascript) <export default as Play>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$film$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Film$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/film.mjs [app-client] (ecmascript) <export default as Film>");
var __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$VideoHUD$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/app/workspace/components/VideoHUD.tsx [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$radar$2f$RadarWidget$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/app/workspace/components/radar/RadarWidget.tsx [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$InsightCard$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/app/workspace/components/InsightCard.tsx [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$reports$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/api/reports.ts [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/apiBase.ts [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$SaveResultsModal$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/app/workspace/components/SaveResultsModal.tsx [app-client] (ecmascript)");
;
var _s = __turbopack_context__.k.signature();
"use client";
;
;
;
;
;
;
;
;
function TacticalDashboard({ job, coachAdvice, coachError, useAltNames = false, dictionary = {}, aiPromptOverride = null, setAiPromptOverride }) {
    _s();
    const [activeIndex, setActiveIndex] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(0);
    const [promptInput, setPromptInput] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])("");
    const [chatHistory, setChatHistory] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])([]);
    const [isTyping, setIsTyping] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(false);
    const [isModalOpen, setIsModalOpen] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(false);
    const [isSaving, setIsSaving] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(false);
    const [saveStatus, setSaveStatus] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])('idle');
    const [llmEngine, setLlmEngine] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])("local");
    const hasAutoPrompted = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(false);
    // Filters State
    const [filterType, setFilterType] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(null);
    const [filterVal, setFilterVal] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(null);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "TacticalDashboard.useEffect": ()=>{
            if ("TURBOPACK compile-time falsy", 0) //TURBOPACK unreachable
            ;
            const syncEngine = {
                "TacticalDashboard.useEffect.syncEngine": ()=>{
                    const pref = localStorage.getItem("gaffer-engine-type") === "cloud" ? "cloud" : "local";
                    setLlmEngine(pref);
                }
            }["TacticalDashboard.useEffect.syncEngine"];
            syncEngine();
            window.addEventListener("gaffer-engine-changed", syncEngine);
            return ({
                "TacticalDashboard.useEffect": ()=>window.removeEventListener("gaffer-engine-changed", syncEngine)
            })["TacticalDashboard.useEffect"];
        }
    }["TacticalDashboard.useEffect"], []);
    // Sync prompts and seek commands from external views
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "TacticalDashboard.useEffect": ()=>{
            if (aiPromptOverride) {
                setPromptInput(aiPromptOverride);
                setAiPromptOverride?.(null);
                setTimeout({
                    "TacticalDashboard.useEffect": ()=>{
                        const botElement = document.getElementById("ai-chat-assistant");
                        if (botElement) {
                            botElement.scrollIntoView({
                                behavior: 'smooth'
                            });
                        }
                    }
                }["TacticalDashboard.useEffect"], 200);
            }
        }
    }["TacticalDashboard.useEffect"], [
        aiPromptOverride,
        setAiPromptOverride
    ]);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "TacticalDashboard.useEffect": ()=>{
            const onPlayClipEvent = {
                "TacticalDashboard.useEffect.onPlayClipEvent": (e)=>{
                    const detail = e.detail;
                    if (detail && typeof detail.startTimeS === 'number') {
                        handlePlayClip(detail.startTimeS);
                    }
                }
            }["TacticalDashboard.useEffect.onPlayClipEvent"];
            window.addEventListener("gaffer-play-clip", onPlayClipEvent);
            return ({
                "TacticalDashboard.useEffect": ()=>window.removeEventListener("gaffer-play-clip", onPlayClipEvent)
            })["TacticalDashboard.useEffect"];
        }
    }["TacticalDashboard.useEffect"], []);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "TacticalDashboard.useEffect": ()=>{
            // Automatically prompt to save when the analysis fully completes
            if (coachAdvice?.advice_items?.length > 0 && !hasAutoPrompted.current && !job?.isHistorical) {
                hasAutoPrompted.current = true;
                setIsModalOpen(true);
            }
        }
    }["TacticalDashboard.useEffect"], [
        coachAdvice,
        job
    ]);
    const videoRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    const scrollRef = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(null);
    const handlePlayClip = (startTimeS)=>{
        if (videoRef.current) {
            videoRef.current.currentTime = startTimeS;
            void videoRef.current.play();
        }
    };
    // Robust video source binding
    const videoSrc = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "TacticalDashboard.useMemo[videoSrc]": ()=>{
            if (job?.file) {
                return URL.createObjectURL(job.file);
            }
            return job?.videoUrl || job?.video_url || "";
        }
    }["TacticalDashboard.useMemo[videoSrc]"], [
        job
    ]);
    // Parse the evidence string when summary_data is null
    // Evidence format: "Tactical Power: Red X vs Blue Y. Win Probability: Red X% | Blue Y%..."
    const parseEvidenceMetrics = (evidence)=>{
        const parsed = {
            team_0: {},
            team_1: {},
            win_probability: {}
        };
        const tp = evidence.match(/Tactical Power: Red ([\d.]+) vs Blue ([\d.]+)/);
        if (tp) {
            parsed.team_0.tactical_power = parseFloat(tp[1]);
            parsed.team_1.tactical_power = parseFloat(tp[2]);
            parsed.team_red_score = parseFloat(tp[1]);
            parsed.team_blue_score = parseFloat(tp[2]);
        }
        const wp = evidence.match(/Win Probability: Red ([\d.]+)%[^|]*\| Blue ([\d.]+)%/);
        if (wp) {
            parsed.win_probability.team_red = parseFloat(wp[1]);
            parsed.win_probability.team_blue = parseFloat(wp[2]);
            parsed.team_0.win_prob = parseFloat(wp[1]);
            parsed.team_1.win_prob = parseFloat(wp[2]);
        }
        const cp = evidence.match(/Compactness: Red ([\d.]+) \/ Blue ([\d.]+)/);
        if (cp) {
            parsed.team_0.compactness = parseFloat(cp[1]);
            parsed.team_1.compactness = parseFloat(cp[2]);
        }
        const ts = evidence.match(/Transition Speed: Red ([\d.]+) \/ Blue ([\d.]+)/);
        if (ts) {
            parsed.team_0.transition_speed = parseFloat(ts[1]);
            parsed.team_1.transition_speed = parseFloat(ts[2]);
        }
        return parsed;
    };
    // Global summary metrics — prefer summary_data, fall back to parsing evidence string
    const globalMetrics = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "TacticalDashboard.useMemo[globalMetrics]": ()=>{
            const summaryItem = coachAdvice?.advice_items?.find({
                "TacticalDashboard.useMemo[globalMetrics]": (i)=>i.flaw === 'Match Summary'
            }["TacticalDashboard.useMemo[globalMetrics]"]);
            if (!summaryItem) return {};
            if (summaryItem.summary_data?.team_0) return summaryItem.summary_data;
            return parseEvidenceMetrics(summaryItem.evidence || '');
        }
    }["TacticalDashboard.useMemo[globalMetrics]"], [
        coachAdvice
    ]);
    const timeline = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "TacticalDashboard.useMemo[timeline]": ()=>{
            if (!coachAdvice?.advice_items) return [];
            const summaryItem = coachAdvice.advice_items.find({
                "TacticalDashboard.useMemo[timeline].summaryItem": (i)=>i.flaw === 'Match Summary'
            }["TacticalDashboard.useMemo[timeline].summaryItem"]);
            const summaryMetrics = summaryItem?.summary_data?.team_0 ? summaryItem.summary_data : parseEvidenceMetrics(summaryItem?.evidence || '');
            return coachAdvice.advice_items.map({
                "TacticalDashboard.useMemo[timeline]": (item)=>({
                        time: item.timestamp || (item.frame_idx !== undefined ? `F:${item.frame_idx}` : "00:00"),
                        title: item.flaw || "Tactical Phase",
                        summary: item.tactical_instruction || item.evidence || "System analysis in progress...",
                        minute: item.minute || "0",
                        // blue = team_1, red = team_0 (consistent with InsightCard teamMetrics mapping)
                        blueTeam: {
                            payload: item.flaw === 'Match Summary' ? item.evidence : item.team === 'team_1' ? item.tactical_instruction : item.evidence || "",
                            keywords: []
                        },
                        redTeam: {
                            payload: item.flaw === 'Match Summary' ? item.evidence : item.team === 'team_0' ? item.tactical_instruction : item.evidence || "",
                            keywords: []
                        },
                        // Always use global summary metrics so numbers are never null
                        metrics: summaryMetrics,
                        evidenceClips: item.evidence_clips || [],
                        threatContext: item.threat_context || null,
                        eventCountSummary: item.event_count_summary || null
                    })
            }["TacticalDashboard.useMemo[timeline]"]);
        }
    }["TacticalDashboard.useMemo[timeline]"], [
        coachAdvice
    ]);
    // Compute filtered timeline
    const filteredTimeline = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "TacticalDashboard.useMemo[filteredTimeline]": ()=>{
            if (!filterType || !filterVal) return timeline;
            return timeline.filter({
                "TacticalDashboard.useMemo[filteredTimeline]": (item)=>{
                    if (filterType === 'category') {
                        const textToSearch = `${item.title} ${item.summary}`.toLowerCase();
                        return textToSearch.includes(filterVal.toLowerCase()) || item.title.toLowerCase().includes(filterVal.toLowerCase());
                    }
                    if (filterType === 'player') {
                        const pidStr = filterVal;
                        const playerRemap = dictionary[`P${pidStr}`] || `Player ${pidStr}`;
                        const textToSearch = `${item.title} ${item.summary}`.toLowerCase();
                        const inText = textToSearch.includes(`player ${pidStr}`) || textToSearch.includes(`p${pidStr}`) || textToSearch.includes(playerRemap.toLowerCase());
                        const inThreat = item.threatContext?.top_threats?.some({
                            "TacticalDashboard.useMemo[filteredTimeline]": (t)=>t.player_id.toString() === pidStr
                        }["TacticalDashboard.useMemo[filteredTimeline]"]);
                        const inClips = item.evidenceClips?.some({
                            "TacticalDashboard.useMemo[filteredTimeline]": (c)=>c.highlight_player_ids?.includes(parseInt(pidStr))
                        }["TacticalDashboard.useMemo[filteredTimeline]"]);
                        return inText || inThreat || inClips;
                    }
                    return true;
                }
            }["TacticalDashboard.useMemo[filteredTimeline]"]);
        }
    }["TacticalDashboard.useMemo[filteredTimeline]"], [
        timeline,
        filterType,
        filterVal,
        dictionary
    ]);
    const activeSlide = timeline[activeIndex];
    const handleSaveBoth = async ()=>{
        if (!coachAdvice) return;
        setIsSaving(true);
        try {
            // 1. Save the report to the backend
            await (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$reports$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["saveTacticalReport"])(job.jobId, coachAdvice, job?.file?.name);
            // 2. Download the video using fetch so we can send auth headers
            setSaveStatus('rendering');
            const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
            const downloadUrl = `${base}/api/v1/elite/jobs/${job.jobId}/video/download`;
            const response = await fetch(downloadUrl, {
                headers: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
            });
            if (!response.ok) {
                throw new Error(`Video download failed: ${response.status}`);
            }
            const blob = await response.blob();
            const blobUrl = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = blobUrl;
            link.setAttribute("download", `GaffersGuide_TacticalRadar_${job.jobId}.mp4`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            setTimeout(()=>URL.revokeObjectURL(blobUrl), 5000);
            setSaveStatus('success');
            setTimeout(()=>{
                setSaveStatus('idle');
                setIsModalOpen(false);
            }, 2000);
        } catch (err) {
            console.error('Save both failed:', err);
            setSaveStatus('error');
            setTimeout(()=>setSaveStatus('idle'), 3000);
        } finally{
            setIsSaving(false);
        }
    };
    const handleSaveReportOnly = async ()=>{
        if (!coachAdvice) return;
        setIsSaving(true);
        try {
            await (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$reports$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["saveTacticalReport"])(job.jobId, coachAdvice, job?.file?.name);
            setSaveStatus('success');
            setTimeout(()=>{
                setSaveStatus('idle');
                setIsModalOpen(false);
            }, 2000);
        } catch (err) {
            setSaveStatus('error');
            setTimeout(()=>setSaveStatus('idle'), 3000);
        } finally{
            setIsSaving(false);
        }
    };
    const handleSendPrompt = async (e)=>{
        e.preventDefault();
        if (!promptInput.trim()) return;
        const userMsg = {
            id: Date.now(),
            role: "user",
            text: promptInput
        };
        setChatHistory((prev)=>[
                ...prev,
                userMsg
            ]);
        let finalMessage = promptInput;
        if (useAltNames && Object.keys(dictionary).length > 0) {
            const translationNotes = Object.entries(dictionary).filter(([_, value])=>value && value.trim()).map(([key, value])=>`* "${value}" means "${key}"`).join("\n");
            if (translationNotes) {
                finalMessage += `\n\n[Coaching Dictionary / Alt Name Translations]:\n${translationNotes}`;
            }
        }
        setPromptInput("");
        setIsTyping(true);
        try {
            const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
            const response = await fetch(`${base}/api/v1/chat`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    ...(0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
                },
                body: JSON.stringify({
                    message: finalMessage,
                    job_id: job?.jobId,
                    llm_engine: llmEngine,
                    context: {
                        flaw: activeSlide?.title,
                        analysis: activeSlide?.summary,
                        match_summary: coachAdvice?.advice_items?.find((i)=>i.flaw === 'Match Summary')?.evidence
                    }
                })
            });
            if (!response.ok) {
                throw new Error(`Chat API failed: ${response.status}`);
            }
            const data = await response.json();
            setChatHistory((prev)=>[
                    ...prev,
                    {
                        id: Date.now() + 1,
                        role: "assistant",
                        text: data.reply || data.response || "No response from tactical engine.",
                        evidence: data.evidence || null
                    }
                ]);
        } catch (err) {
            console.error(err);
            setChatHistory((prev)=>[
                    ...prev,
                    {
                        id: Date.now() + 1,
                        role: "assistant",
                        text: "Tactical engine temporarily offline. Verify backend connectivity."
                    }
                ]);
        } finally{
            setIsTyping(false);
        }
    };
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "TacticalDashboard.useEffect": ()=>{
            if (scrollRef.current) {
                scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
            }
        }
    }["TacticalDashboard.useEffect"], [
        chatHistory
    ]);
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "flex flex-col h-full w-full bg-[#050805] font-sans text-gray-200 overflow-y-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:bg-gray-800",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "grid grid-cols-[0.9fr_2fr_1.8fr] h-[500px] min-h-[500px] w-full gap-4 p-4",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex flex-col bg-[#0a0f0a] border border-gray-900 rounded-xl overflow-hidden shadow-2xl",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "p-4 border-b border-gray-900 flex justify-between items-center bg-[#111a12]/20",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                                        className: "text-[10px] font-bold font-mono uppercase tracking-[0.3em] text-gray-500",
                                        children: "Match Events"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                        lineNumber: 360,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                        onClick: ()=>setIsModalOpen(true),
                                        className: "text-gray-500 hover:text-emerald-500 transition-all p-1 hover:bg-emerald-500/10 rounded",
                                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$save$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Save$3e$__["Save"], {
                                            size: 14
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                            lineNumber: 362,
                                            columnNumber: 15
                                        }, this)
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                        lineNumber: 361,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                lineNumber: 359,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "p-3 border-b border-gray-900 bg-black/10 space-y-2",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "flex items-center justify-between text-[9px] font-mono text-gray-500",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "uppercase tracking-widest font-bold",
                                                children: "Filters & Toggles"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                lineNumber: 369,
                                                columnNumber: 15
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                onClick: ()=>{
                                                    setFilterType(null);
                                                    setFilterVal(null);
                                                },
                                                className: "text-[9px] hover:text-emerald-400 underline transition-colors",
                                                children: "Clear"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                lineNumber: 370,
                                                columnNumber: 15
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                        lineNumber: 368,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "flex gap-2 text-[9px] font-mono",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: "flex items-center gap-1 text-gray-500",
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                        className: `h-1.5 w-1.5 rounded-full ${useAltNames ? 'bg-emerald-500' : 'bg-gray-800'}`
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                        lineNumber: 383,
                                                        columnNumber: 17
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                        children: "Alt Names"
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                        lineNumber: 384,
                                                        columnNumber: 17
                                                    }, this)
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                lineNumber: 382,
                                                columnNumber: 15
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: "flex items-center gap-1 text-gray-500",
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                        className: "h-1.5 w-1.5 rounded-full bg-emerald-500"
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                        lineNumber: 387,
                                                        columnNumber: 17
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                        children: "Analysis Data"
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                        lineNumber: 388,
                                                        columnNumber: 17
                                                    }, this)
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                lineNumber: 386,
                                                columnNumber: 15
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                        lineNumber: 381,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "flex flex-wrap gap-1 pt-1 max-h-[85px] overflow-y-auto [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-thumb]:bg-gray-800",
                                        children: [
                                            [
                                                'Goal',
                                                'Shot',
                                                'Defense',
                                                'Recovery',
                                                'Possession'
                                            ].map((cat)=>{
                                                const isSelected = filterType === 'category' && filterVal === cat;
                                                return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                    onClick: ()=>{
                                                        setFilterType(isSelected ? null : 'category');
                                                        setFilterVal(isSelected ? null : cat);
                                                    },
                                                    className: `px-2 py-0.5 rounded text-[8px] font-mono uppercase font-bold transition-all border ${isSelected ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' : 'bg-transparent text-gray-600 border-gray-900 hover:text-gray-400 hover:border-gray-800'}`,
                                                    children: useAltNames && dictionary[cat] ? dictionary[cat] : cat
                                                }, cat, false, {
                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                    lineNumber: 396,
                                                    columnNumber: 19
                                                }, this);
                                            }),
                                            [
                                                1,
                                                2,
                                                3,
                                                4,
                                                5,
                                                6,
                                                7,
                                                8
                                            ].map((pid)=>{
                                                const playerLabel = useAltNames && dictionary[`P${pid}`] ? dictionary[`P${pid}`] : `P${pid}`;
                                                const isSelected = filterType === 'player' && filterVal === pid.toString();
                                                return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                    onClick: ()=>{
                                                        setFilterType(isSelected ? null : 'player');
                                                        setFilterVal(isSelected ? null : pid.toString());
                                                    },
                                                    className: `px-2 py-0.5 rounded text-[8px] font-mono uppercase font-bold transition-all border ${isSelected ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' : 'bg-transparent text-gray-600 border-gray-900 hover:text-gray-400 hover:border-gray-800'}`,
                                                    children: playerLabel
                                                }, pid, false, {
                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                    lineNumber: 417,
                                                    columnNumber: 19
                                                }, this);
                                            })
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                        lineNumber: 392,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                lineNumber: 367,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex-1 overflow-y-auto p-3 space-y-2 [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-thumb]:bg-gray-800",
                                children: filteredTimeline.map((data, i)=>{
                                    const originalIndex = timeline.findIndex((t)=>t.title === data.title && t.time === data.time);
                                    const isCurrentActive = originalIndex === activeIndex;
                                    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                        onClick: ()=>setActiveIndex(originalIndex),
                                        className: `w-full p-4 text-left rounded-xl transition-all border ${isCurrentActive ? "border-emerald-500/50 bg-emerald-500/10 shadow-[0_0_20px_rgba(16,185,129,0.05)]" : "border-gray-900/50 bg-transparent hover:bg-gray-900/50"}`,
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: "flex items-center gap-2 mb-1.5",
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$clock$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Clock$3e$__["Clock"], {
                                                        size: 10,
                                                        className: isCurrentActive ? 'text-emerald-500' : 'text-gray-600'
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                        lineNumber: 449,
                                                        columnNumber: 21
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                        className: `text-[10px] font-mono ${isCurrentActive ? 'text-emerald-500 font-bold' : 'text-gray-600'}`,
                                                        children: data.time
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                        lineNumber: 450,
                                                        columnNumber: 21
                                                    }, this)
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                lineNumber: 448,
                                                columnNumber: 19
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: `text-xs font-bold tracking-tight ${isCurrentActive ? 'text-white' : 'text-gray-500'}`,
                                                children: data.title
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                lineNumber: 452,
                                                columnNumber: 19
                                            }, this)
                                        ]
                                    }, i, true, {
                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                        lineNumber: 441,
                                        columnNumber: 17
                                    }, this);
                                })
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                lineNumber: 436,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                        lineNumber: 358,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex flex-col bg-[#0a0f0a] border border-gray-900 rounded-xl overflow-hidden shadow-2xl group",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "p-3 border-b border-gray-900 flex justify-between items-center bg-[#111a12]/10",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "flex items-center gap-2",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: "h-2 w-2 rounded-full bg-red-600 animate-pulse"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                lineNumber: 463,
                                                columnNumber: 15
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "text-[10px] font-bold font-mono uppercase tracking-widest text-gray-500",
                                                children: "Live Telemetry Feed"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                lineNumber: 464,
                                                columnNumber: 15
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                        lineNumber: 462,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$maximize$2d$2$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Maximize2$3e$__["Maximize2"], {
                                        size: 12,
                                        className: "text-gray-700 hover:text-gray-400 cursor-pointer"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                        lineNumber: 466,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                lineNumber: 461,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex-1 min-h-0 bg-black",
                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$VideoHUD$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"], {
                                    videoRef: videoRef,
                                    videoSrc: videoSrc,
                                    jobId: job?.jobId ?? null,
                                    status: job?.tracking ? "completed" : "pending",
                                    onDownload: ()=>{},
                                    trackingData: job?.tracking ?? null,
                                    useAltNames: useAltNames,
                                    dictionary: dictionary
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                    lineNumber: 469,
                                    columnNumber: 14
                                }, this)
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                lineNumber: 468,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                        lineNumber: 460,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex flex-col bg-[#0a0f0a] border border-gray-900 rounded-xl overflow-hidden shadow-2xl",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "p-3 border-b border-gray-900 flex justify-between items-center bg-[#111a12]/10",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "flex items-center gap-2",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$zap$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Zap$3e$__["Zap"], {
                                                size: 12,
                                                className: "text-emerald-500"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                lineNumber: 486,
                                                columnNumber: 15
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "text-[10px] font-bold font-mono uppercase tracking-widest text-gray-500",
                                                children: "Expanded Tactical Radar"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                lineNumber: 487,
                                                columnNumber: 15
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                        lineNumber: 485,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "text-[10px] font-mono text-emerald-500/50",
                                        children: "105m x 68m"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                        lineNumber: 489,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                lineNumber: 484,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex-1 min-h-0",
                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$radar$2f$RadarWidget$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"], {
                                    videoRef: videoRef,
                                    trackingData: job?.tracking ?? null
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                    lineNumber: 492,
                                    columnNumber: 14
                                }, this)
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                lineNumber: 491,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                        lineNumber: 483,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                lineNumber: 355,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "grid grid-cols-4 gap-4 px-4 pb-4",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "bg-[#0a0f0a] border border-gray-900 rounded-xl p-6 flex flex-col gap-2 shadow-lg",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex items-center gap-2 text-[10px] font-bold font-mono uppercase text-gray-600 tracking-widest",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$activity$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Activity$3e$__["Activity"], {
                                        size: 14,
                                        className: "text-emerald-500"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                        lineNumber: 501,
                                        columnNumber: 13
                                    }, this),
                                    "Tactical Power"
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                lineNumber: 500,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "text-3xl font-bold text-emerald-400 tracking-tighter",
                                children: [
                                    globalMetrics?.team_0?.tactical_power?.toFixed(1) ?? globalMetrics?.team_red_score?.toFixed(1) ?? "—",
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "text-xs text-gray-500 ml-1",
                                        children: [
                                            "/ ",
                                            globalMetrics?.team_1?.tactical_power?.toFixed(1) ?? globalMetrics?.team_blue_score?.toFixed(1) ?? "—"
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                        lineNumber: 506,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                lineNumber: 504,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "text-[9px] font-mono text-gray-600",
                                children: "Red / Blue (composite)"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                lineNumber: 508,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                        lineNumber: 499,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "bg-[#0a0f0a] border border-gray-900 rounded-xl p-6 flex flex-col gap-2 shadow-lg",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex items-center gap-2 text-[10px] font-bold font-mono uppercase text-gray-600 tracking-widest",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$trending$2d$up$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__TrendingUp$3e$__["TrendingUp"], {
                                        size: 14,
                                        className: "text-cyan-500"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                        lineNumber: 512,
                                        columnNumber: 13
                                    }, this),
                                    "Win Probability"
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                lineNumber: 511,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "text-3xl font-bold text-cyan-400 tracking-tighter",
                                children: [
                                    globalMetrics?.win_probability?.team_red ?? globalMetrics?.team_0?.win_prob ?? "—",
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "text-xs text-gray-600 ml-1",
                                        children: "%"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                        lineNumber: 517,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                lineNumber: 515,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                        lineNumber: 510,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "bg-[#0a0f0a] border border-gray-900 rounded-xl p-6 flex flex-col gap-2 shadow-lg",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex items-center gap-2 text-[10px] font-bold font-mono uppercase text-gray-600 tracking-widest",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$shield$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Shield$3e$__["Shield"], {
                                        size: 14,
                                        className: "text-amber-500"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                        lineNumber: 522,
                                        columnNumber: 13
                                    }, this),
                                    "Compactness"
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                lineNumber: 521,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "text-3xl font-bold text-amber-400 tracking-tighter",
                                children: globalMetrics?.team_0?.compactness != null ? globalMetrics.team_0.compactness > 70 ? "High" : globalMetrics.team_0.compactness > 40 ? "Med" : "Low" : "—"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                lineNumber: 525,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                        lineNumber: 520,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "bg-[#0a0f0a] border border-gray-900 rounded-xl p-6 flex flex-col gap-2 shadow-lg",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex items-center gap-2 text-[10px] font-bold font-mono uppercase text-gray-600 tracking-widest",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$zap$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Zap$3e$__["Zap"], {
                                        size: 14,
                                        className: "text-purple-500"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                        lineNumber: 533,
                                        columnNumber: 13
                                    }, this),
                                    "Transition Speed"
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                lineNumber: 532,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "text-3xl font-bold text-purple-400 tracking-tighter",
                                children: globalMetrics?.team_0?.transition_speed != null ? globalMetrics.team_0.transition_speed > 70 ? "Fast" : globalMetrics.team_0.transition_speed > 40 ? "Norm" : "Slow" : "—"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                lineNumber: 536,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                        lineNumber: 531,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                lineNumber: 498,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "px-4 pb-8",
                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: "bg-[#0a0f0a] border border-gray-900 rounded-2xl p-8 shadow-2xl",
                    children: [
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "mb-6 flex items-center justify-between border-b border-gray-900 pb-4",
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                                    className: "text-xl font-bold tracking-tight text-white",
                                    children: "Match Intelligence Report"
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                    lineNumber: 548,
                                    columnNumber: 13
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "text-[10px] font-mono text-gray-600 uppercase tracking-widest",
                                    children: "Pipeline Analysis Active"
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                    lineNumber: 549,
                                    columnNumber: 13
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                            lineNumber: 547,
                            columnNumber: 11
                        }, this),
                        activeSlide ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$InsightCard$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["InsightCard"], {
                            title: activeSlide.title,
                            minute: activeSlide.minute,
                            blueTeam: activeSlide.blueTeam,
                            redTeam: activeSlide.redTeam,
                            metrics: activeSlide.metrics,
                            evidenceClips: activeSlide.evidenceClips,
                            threatContext: activeSlide.threatContext,
                            eventCountSummary: activeSlide.eventCountSummary,
                            onPlayClip: handlePlayClip,
                            useAltNames: useAltNames,
                            dictionary: dictionary
                        }, activeIndex, false, {
                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                            lineNumber: 553,
                            columnNumber: 13
                        }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "h-48 flex flex-col items-center justify-center gap-2 text-gray-500 text-sm italic font-mono animate-pulse",
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2d$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader2$3e$__["Loader2"], {
                                    className: "animate-spin text-emerald-500",
                                    size: 24
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                    lineNumber: 569,
                                    columnNumber: 15
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                    children: "Synchronizing tactical debrief..."
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                    lineNumber: 570,
                                    columnNumber: 15
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                            lineNumber: 568,
                            columnNumber: 13
                        }, this)
                    ]
                }, void 0, true, {
                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                    lineNumber: 546,
                    columnNumber: 9
                }, this)
            }, void 0, false, {
                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                lineNumber: 545,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                id: "ai-chat-assistant",
                className: "px-4 pb-12",
                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: "bg-[#111a12]/30 border border-emerald-500/20 rounded-2xl overflow-hidden shadow-[0_0_50px_rgba(16,185,129,0.05)]",
                    children: [
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "p-5 border-b border-emerald-500/10 bg-[#111a12]/40 flex items-center justify-between",
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "flex items-center gap-3",
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "p-2 bg-emerald-500/20 rounded-lg",
                                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$bot$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Bot$3e$__["Bot"], {
                                                size: 20,
                                                className: "text-emerald-500"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                lineNumber: 582,
                                                columnNumber: 17
                                            }, this)
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                            lineNumber: 581,
                                            columnNumber: 15
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            children: [
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h3", {
                                                    className: "text-sm font-bold text-white tracking-tight",
                                                    children: "Tactical AI Assistant"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                    lineNumber: 585,
                                                    columnNumber: 17
                                                }, this),
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                                    className: "text-[10px] text-emerald-500/60 font-mono uppercase tracking-widest",
                                                    children: "Match-Specific Follow-Up Engine"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                    lineNumber: 586,
                                                    columnNumber: 17
                                                }, this)
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                            lineNumber: 584,
                                            columnNumber: 15
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                    lineNumber: 580,
                                    columnNumber: 13
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "flex items-center gap-4 text-[10px] font-mono text-gray-600",
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                            children: [
                                                "Engine: ",
                                                llmEngine === 'cloud' ? 'Cloud (Gemini / OpenAI)' : 'Ollama / Llama 3'
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                            lineNumber: 590,
                                            columnNumber: 16
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                            className: "h-1 w-1 bg-gray-800 rounded-full"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                            lineNumber: 591,
                                            columnNumber: 16
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                            children: "Context: Active Phase"
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                            lineNumber: 592,
                                            columnNumber: 16
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                    lineNumber: 589,
                                    columnNumber: 13
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                            lineNumber: 579,
                            columnNumber: 11
                        }, this),
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            className: "h-[400px] flex flex-col relative",
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    ref: scrollRef,
                                    className: "flex-1 overflow-y-auto p-8 space-y-6 [&::-webkit-scrollbar]:w-1 [&::-webkit-scrollbar-thumb]:bg-gray-800 pb-32",
                                    children: [
                                        chatHistory.length === 0 && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "h-full flex flex-col items-center justify-center space-y-4 opacity-30 text-center",
                                            children: [
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$message$2d$square$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__MessageSquare$3e$__["MessageSquare"], {
                                                    size: 48,
                                                    className: "text-gray-600"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                    lineNumber: 600,
                                                    columnNumber: 19
                                                }, this),
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                    className: "space-y-1",
                                                    children: [
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                                            className: "text-sm font-bold text-gray-300",
                                                            children: "Ready for tactical inquiry"
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                            lineNumber: 602,
                                                            columnNumber: 21
                                                        }, this),
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                                            className: "text-xs text-gray-500 max-w-xs",
                                                            children: "Ask about specific player roles, transitional phases, or philosophical adjustments."
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                            lineNumber: 603,
                                                            columnNumber: 21
                                                        }, this)
                                                    ]
                                                }, void 0, true, {
                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                    lineNumber: 601,
                                                    columnNumber: 19
                                                }, this)
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                            lineNumber: 599,
                                            columnNumber: 17
                                        }, this),
                                        chatHistory.map((msg)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: `flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`,
                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                    className: `max-w-[70%] p-5 rounded-2xl shadow-xl ${msg.role === 'user' ? 'bg-[#1a241c] border border-emerald-500/30 text-white rounded-tr-none' : 'bg-[#0a0f0a] border border-gray-800 text-gray-300 rounded-tl-none'}`,
                                                    children: [
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                            className: "text-[10px] font-mono uppercase tracking-widest text-gray-600 mb-2",
                                                            children: msg.role === 'user' ? 'Coach' : 'Gaffer Assistant'
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                            lineNumber: 614,
                                                            columnNumber: 21
                                                        }, this),
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                            className: "text-sm leading-relaxed",
                                                            children: msg.text
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                            lineNumber: 615,
                                                            columnNumber: 21
                                                        }, this),
                                                        msg.evidence && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                            className: "mt-4 pt-4 border-t border-emerald-500/10 space-y-4 font-sans",
                                                            children: [
                                                                msg.evidence.clips && msg.evidence.clips.length > 0 && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                    className: "space-y-2",
                                                                    children: [
                                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                            className: "flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider text-emerald-500/80",
                                                                            children: [
                                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$film$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Film$3e$__["Film"], {
                                                                                    size: 12
                                                                                }, void 0, false, {
                                                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                    lineNumber: 623,
                                                                                    columnNumber: 31
                                                                                }, this),
                                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                                    children: "Supporting Video Evidence"
                                                                                }, void 0, false, {
                                                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                    lineNumber: 624,
                                                                                    columnNumber: 31
                                                                                }, this)
                                                                            ]
                                                                        }, void 0, true, {
                                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                            lineNumber: 622,
                                                                            columnNumber: 29
                                                                        }, this),
                                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                            className: "grid grid-cols-1 gap-2",
                                                                            children: msg.evidence.clips.map((clip, clipIdx)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                                                    onClick: ()=>handlePlayClip(clip.start_time_s),
                                                                                    className: "w-full text-left p-3 rounded-lg bg-black/40 border border-gray-800 hover:border-emerald-500/30 hover:bg-[#111a12]/20 transition-all flex items-center justify-between group",
                                                                                    children: [
                                                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                            className: "flex-1 min-w-0 pr-3",
                                                                                            children: [
                                                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                                    className: "text-xs font-bold text-gray-200 truncate group-hover:text-emerald-400 transition-colors",
                                                                                                    children: clip.label || clip.event_name || "Evidence Clip"
                                                                                                }, void 0, false, {
                                                                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                                    lineNumber: 634,
                                                                                                    columnNumber: 37
                                                                                                }, this),
                                                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                                    className: "text-[9px] font-mono text-gray-500 mt-0.5",
                                                                                                    children: [
                                                                                                        "Timestamp: ",
                                                                                                        Math.floor(clip.start_time_s / 60),
                                                                                                        ":",
                                                                                                        Math.floor(clip.start_time_s % 60).toString().padStart(2, '0'),
                                                                                                        " - ",
                                                                                                        Math.floor(clip.end_time_s / 60),
                                                                                                        ":",
                                                                                                        Math.floor(clip.end_time_s % 60).toString().padStart(2, '0')
                                                                                                    ]
                                                                                                }, void 0, true, {
                                                                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                                    lineNumber: 637,
                                                                                                    columnNumber: 37
                                                                                                }, this)
                                                                                            ]
                                                                                        }, void 0, true, {
                                                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                            lineNumber: 633,
                                                                                            columnNumber: 35
                                                                                        }, this),
                                                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                            className: "h-6 w-6 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-500 group-hover:bg-emerald-500/20 group-hover:scale-105 transition-all",
                                                                                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$play$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Play$3e$__["Play"], {
                                                                                                size: 10,
                                                                                                fill: "currentColor"
                                                                                            }, void 0, false, {
                                                                                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                                lineNumber: 642,
                                                                                                columnNumber: 37
                                                                                            }, this)
                                                                                        }, void 0, false, {
                                                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                            lineNumber: 641,
                                                                                            columnNumber: 35
                                                                                        }, this)
                                                                                    ]
                                                                                }, clipIdx, true, {
                                                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                    lineNumber: 628,
                                                                                    columnNumber: 33
                                                                                }, this))
                                                                        }, void 0, false, {
                                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                            lineNumber: 626,
                                                                            columnNumber: 29
                                                                        }, this)
                                                                    ]
                                                                }, void 0, true, {
                                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                    lineNumber: 621,
                                                                    columnNumber: 27
                                                                }, this),
                                                                msg.evidence.top_threats && msg.evidence.top_threats.length > 0 && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                    className: "space-y-2",
                                                                    children: [
                                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                            className: "flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-wider text-emerald-500/80",
                                                                            children: [
                                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$user$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__User$3e$__["User"], {
                                                                                    size: 12
                                                                                }, void 0, false, {
                                                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                    lineNumber: 654,
                                                                                    columnNumber: 31
                                                                                }, this),
                                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                                    children: "Key Attacking Threats"
                                                                                }, void 0, false, {
                                                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                    lineNumber: 655,
                                                                                    columnNumber: 31
                                                                                }, this)
                                                                            ]
                                                                        }, void 0, true, {
                                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                            lineNumber: 653,
                                                                            columnNumber: 29
                                                                        }, this),
                                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                            className: "grid grid-cols-1 gap-2",
                                                                            children: msg.evidence.top_threats.map((threat, threatIdx)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                    className: "p-3 rounded-lg bg-black/30 border border-gray-800/80 space-y-2",
                                                                                    children: [
                                                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                            className: "flex items-center justify-between",
                                                                                            children: [
                                                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                                    className: "flex items-center gap-2",
                                                                                                    children: [
                                                                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                                            className: `h-5 w-5 rounded-full flex items-center justify-center text-[10px] font-bold ${threat.team_id === 'team_0' ? 'bg-red-500/20 text-red-400' : 'bg-blue-500/20 text-blue-400'}`,
                                                                                                            children: [
                                                                                                                "P",
                                                                                                                threat.player_id
                                                                                                            ]
                                                                                                        }, void 0, true, {
                                                                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                                            lineNumber: 662,
                                                                                                            columnNumber: 39
                                                                                                        }, this),
                                                                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                                                            className: "text-xs font-bold text-gray-300 uppercase",
                                                                                                            children: threat.team_id === 'team_0' ? 'Red Team' : 'Blue Team'
                                                                                                        }, void 0, false, {
                                                                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                                            lineNumber: 665,
                                                                                                            columnNumber: 39
                                                                                                        }, this)
                                                                                                    ]
                                                                                                }, void 0, true, {
                                                                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                                    lineNumber: 661,
                                                                                                    columnNumber: 37
                                                                                                }, this),
                                                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                                    className: "flex items-center gap-1",
                                                                                                    children: [
                                                                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                                                            className: "text-[10px] font-mono text-gray-500",
                                                                                                            children: "Threat:"
                                                                                                        }, void 0, false, {
                                                                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                                            lineNumber: 670,
                                                                                                            columnNumber: 39
                                                                                                        }, this),
                                                                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                                                            className: "text-xs font-bold text-emerald-400 font-mono",
                                                                                                            children: threat.threat_score.toFixed(1)
                                                                                                        }, void 0, false, {
                                                                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                                            lineNumber: 671,
                                                                                                            columnNumber: 39
                                                                                                        }, this)
                                                                                                    ]
                                                                                                }, void 0, true, {
                                                                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                                    lineNumber: 669,
                                                                                                    columnNumber: 37
                                                                                                }, this)
                                                                                            ]
                                                                                        }, void 0, true, {
                                                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                            lineNumber: 660,
                                                                                            columnNumber: 35
                                                                                        }, this),
                                                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                            className: "h-1 bg-gray-900 rounded-full overflow-hidden",
                                                                                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                                className: `h-full ${threat.team_id === 'team_0' ? 'bg-red-500' : 'bg-blue-500'}`,
                                                                                                style: {
                                                                                                    width: `${threat.threat_score}%`
                                                                                                }
                                                                                            }, void 0, false, {
                                                                                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                                lineNumber: 677,
                                                                                                columnNumber: 37
                                                                                            }, this)
                                                                                        }, void 0, false, {
                                                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                            lineNumber: 676,
                                                                                            columnNumber: 35
                                                                                        }, this),
                                                                                        threat.explanation && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                                                                            className: "text-[10px] text-gray-400 leading-normal font-sans",
                                                                                            children: threat.explanation
                                                                                        }, void 0, false, {
                                                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                            lineNumber: 684,
                                                                                            columnNumber: 37
                                                                                        }, this)
                                                                                    ]
                                                                                }, threatIdx, true, {
                                                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                                    lineNumber: 659,
                                                                                    columnNumber: 33
                                                                                }, this))
                                                                        }, void 0, false, {
                                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                            lineNumber: 657,
                                                                            columnNumber: 29
                                                                        }, this)
                                                                    ]
                                                                }, void 0, true, {
                                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                                    lineNumber: 652,
                                                                    columnNumber: 27
                                                                }, this)
                                                            ]
                                                        }, void 0, true, {
                                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                            lineNumber: 618,
                                                            columnNumber: 23
                                                        }, this)
                                                    ]
                                                }, void 0, true, {
                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                    lineNumber: 609,
                                                    columnNumber: 19
                                                }, this)
                                            }, msg.id, false, {
                                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                lineNumber: 608,
                                                columnNumber: 17
                                            }, this)),
                                        isTyping && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                            className: "flex gap-4 items-center animate-pulse",
                                            children: [
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                    className: "h-8 w-8 bg-emerald-500/20 rounded-full flex items-center justify-center",
                                                    children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$bot$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Bot$3e$__["Bot"], {
                                                        size: 14,
                                                        className: "text-emerald-500"
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                        lineNumber: 701,
                                                        columnNumber: 21
                                                    }, this)
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                    lineNumber: 700,
                                                    columnNumber: 19
                                                }, this),
                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                    className: "h-2 w-24 bg-gray-800 rounded-full"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                    lineNumber: 703,
                                                    columnNumber: 19
                                                }, this)
                                            ]
                                        }, void 0, true, {
                                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                            lineNumber: 699,
                                            columnNumber: 17
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                    lineNumber: 597,
                                    columnNumber: 13
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "absolute bottom-0 left-0 right-0 p-8 bg-gradient-to-t from-[#050805] via-[#050805] to-transparent",
                                    children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("form", {
                                        onSubmit: handleSendPrompt,
                                        className: "relative mx-auto max-w-4xl",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                                type: "text",
                                                value: promptInput,
                                                onChange: (e)=>setPromptInput(e.target.value),
                                                placeholder: "Inquire about transitional overload or defensive compactness...",
                                                className: "w-full bg-[#111a12] border border-emerald-500/20 rounded-2xl py-5 pl-8 pr-16 text-sm text-white focus:border-emerald-500/50 focus:ring-2 focus:ring-emerald-500/10 outline-none transition-all placeholder:text-gray-700 shadow-2xl"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                lineNumber: 710,
                                                columnNumber: 17
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                type: "submit",
                                                className: "absolute right-4 top-1/2 -translate-y-1/2 p-3 bg-emerald-500 text-black rounded-xl hover:bg-emerald-400 transition-all shadow-[0_0_20px_rgba(16,185,129,0.3)]",
                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$send$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Send$3e$__["Send"], {
                                                    size: 18
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                    lineNumber: 718,
                                                    columnNumber: 19
                                                }, this)
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                                lineNumber: 717,
                                                columnNumber: 17
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                        lineNumber: 709,
                                        columnNumber: 15
                                    }, this)
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                                    lineNumber: 708,
                                    columnNumber: 13
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                            lineNumber: 596,
                            columnNumber: 11
                        }, this)
                    ]
                }, void 0, true, {
                    fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                    lineNumber: 578,
                    columnNumber: 9
                }, this)
            }, void 0, false, {
                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                lineNumber: 577,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$SaveResultsModal$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["SaveResultsModal"], {
                isOpen: isModalOpen,
                onClose: ()=>setIsModalOpen(false),
                onSaveBoth: handleSaveBoth,
                onSaveReportOnly: handleSaveReportOnly,
                isSaving: isSaving,
                saveStatus: saveStatus
            }, void 0, false, {
                fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
                lineNumber: 726,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/app/workspace/components/TacticalDashboard.tsx",
        lineNumber: 352,
        columnNumber: 5
    }, this);
}
_s(TacticalDashboard, "HrbVauCkfk5EM7bnVEjSc+darhA=");
_c = TacticalDashboard;
var _c;
__turbopack_context__.k.register(_c, "TacticalDashboard");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/app/workspace/components/ReportsArchive.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "ReportsArchive",
    ()=>ReportsArchive
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$database$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Database$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/database.mjs [app-client] (ecmascript) <export default as Database>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$search$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Search$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/search.mjs [app-client] (ecmascript) <export default as Search>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$folder$2d$open$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__FolderOpen$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/folder-open.mjs [app-client] (ecmascript) <export default as FolderOpen>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$calendar$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Calendar$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/calendar.mjs [app-client] (ecmascript) <export default as Calendar>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$arrow$2d$right$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__ArrowRight$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/arrow-right.mjs [app-client] (ecmascript) <export default as ArrowRight>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2d$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader2$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/loader-circle.mjs [app-client] (ecmascript) <export default as Loader2>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$trash$2d$2$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Trash2$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/trash-2.mjs [app-client] (ecmascript) <export default as Trash2>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$film$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Film$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/film.mjs [app-client] (ecmascript) <export default as Film>");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$jobs$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/api/jobs.ts [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/apiBase.ts [app-client] (ecmascript)");
;
var _s = __turbopack_context__.k.signature();
"use client";
;
;
;
;
function ReportsArchive({ onOpenReport }) {
    _s();
    const [reports, setReports] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])([]);
    const [loading, setLoading] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(true);
    const [searchTerm, setSearchTerm] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])('');
    const [downloadingVideo, setDownloadingVideo] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(null);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "ReportsArchive.useEffect": ()=>{
            loadReports();
        }
    }["ReportsArchive.useEffect"], []);
    async function loadReports() {
        try {
            const data = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$jobs$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["listReports"])();
            setReports(data);
        } catch (error) {
            console.error("Failed to load reports:", error);
        } finally{
            setLoading(false);
        }
    }
    async function handleOpen(reportId) {
        try {
            const fullReport = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$jobs$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getReport"])(reportId);
            onOpenReport(fullReport);
        } catch (error) {
            console.error("Failed to open report:", error);
        }
    }
    async function handleDelete(e, reportId) {
        e.stopPropagation(); // Prevent opening the report when clicking delete
        if (!window.confirm("Are you sure you want to delete this report? This cannot be undone.")) {
            return;
        }
        try {
            await (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$jobs$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["deleteReport"])(reportId);
            // Refresh the list after successful deletion
            await loadReports();
        } catch (error) {
            console.error("Failed to delete report:", error);
            alert("Failed to delete the report. Please try again.");
        }
    }
    async function handleDownloadVideo(e, jobId) {
        e.stopPropagation(); // Prevent opening the report
        if (!jobId || jobId === 'unknown' || jobId === 'manual') {
            alert("No video is associated with this report.");
            return;
        }
        setDownloadingVideo(jobId);
        try {
            const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
            const downloadUrl = `${base}/api/v1/elite/jobs/${jobId}/video/download`;
            const response = await fetch(downloadUrl, {
                headers: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
            });
            if (!response.ok) {
                throw new Error(`Video download failed: ${response.status}`);
            }
            const blob = await response.blob();
            const blobUrl = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = blobUrl;
            link.setAttribute("download", `GaffersGuide_TacticalRadar_${jobId}.mp4`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            setTimeout(()=>URL.revokeObjectURL(blobUrl), 5000);
        } catch (error) {
            console.error("Failed to download video:", error);
            alert("Failed to download the video. The video may not have been generated for this analysis.");
        } finally{
            setDownloadingVideo(null);
        }
    }
    const filteredReports = reports.filter((r)=>(r.video_title || '').toLowerCase().includes(searchTerm.toLowerCase()) || (r.quality_profile || '').toLowerCase().includes(searchTerm.toLowerCase()));
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "h-full w-full bg-[#0a0f0a] flex flex-col animate-fade-in font-sans p-8",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "flex items-center justify-between mb-8",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h1", {
                                className: "text-2xl font-bold text-gray-200 tracking-tight flex items-center gap-3",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$folder$2d$open$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__FolderOpen$3e$__["FolderOpen"], {
                                        className: "text-emerald-500"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                        lineNumber: 98,
                                        columnNumber: 13
                                    }, this),
                                    "Tactical Intelligence Repository"
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                lineNumber: 97,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                className: "text-sm text-gray-500 mt-2 font-mono",
                                children: "Persistent tactical reports and historical match analytics."
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                lineNumber: 101,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                        lineNumber: 96,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "relative",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                type: "text",
                                placeholder: "Search reports...",
                                value: searchTerm,
                                onChange: (e)=>setSearchTerm(e.target.value),
                                className: "bg-[#111a12] border border-gray-800 rounded-lg pl-10 pr-4 py-2 text-sm text-gray-300 focus:outline-none focus:border-emerald-500 transition-colors w-64"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                lineNumber: 105,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$search$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Search$3e$__["Search"], {
                                size: 16,
                                className: "absolute left-3 top-1/2 -translate-y-1/2 text-gray-600"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                lineNumber: 112,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                        lineNumber: 104,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                lineNumber: 95,
                columnNumber: 7
            }, this),
            loading ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "flex-1 flex items-center justify-center",
                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2d$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader2$3e$__["Loader2"], {
                    className: "animate-spin text-emerald-500",
                    size: 48
                }, void 0, false, {
                    fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                    lineNumber: 118,
                    columnNumber: 11
                }, this)
            }, void 0, false, {
                fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                lineNumber: 117,
                columnNumber: 9
            }, this) : filteredReports.length === 0 ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "flex-1 flex flex-col items-center justify-center border-2 border-dashed border-gray-900 rounded-2xl",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$database$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Database$3e$__["Database"], {
                        className: "text-gray-800 mb-4",
                        size: 64
                    }, void 0, false, {
                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                        lineNumber: 122,
                        columnNumber: 11
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                        className: "text-gray-600 font-mono",
                        children: "No tactical reports found in repository."
                    }, void 0, false, {
                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                        lineNumber: 123,
                        columnNumber: 11
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                lineNumber: 121,
                columnNumber: 9
            }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 overflow-y-auto pr-2 custom-scrollbar",
                children: filteredReports.map((report)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        onClick: ()=>handleOpen(report.id),
                        className: "bg-[#111a12] border border-gray-900 hover:border-emerald-500/50 rounded-xl p-5 cursor-pointer transition-all hover:bg-[#152018] group flex flex-col justify-between min-h-[240px] shadow-sm hover:shadow-emerald-900/10",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "flex justify-between items-start mb-4",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: "p-2 bg-emerald-500/10 rounded-lg group-hover:bg-emerald-500/20 transition-colors",
                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$database$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Database$3e$__["Database"], {
                                                    size: 20,
                                                    className: "text-emerald-500"
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                                    lineNumber: 136,
                                                    columnNumber: 21
                                                }, this)
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                                lineNumber: 135,
                                                columnNumber: 19
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: "flex items-center gap-1",
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                        className: "text-[10px] font-bold text-gray-600 font-mono flex items-center gap-1 uppercase tracking-tighter",
                                                        children: [
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$calendar$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Calendar$3e$__["Calendar"], {
                                                                size: 12
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                                                lineNumber: 140,
                                                                columnNumber: 23
                                                            }, this),
                                                            " ",
                                                            new Date(report.saved_at).toLocaleDateString()
                                                        ]
                                                    }, void 0, true, {
                                                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                                        lineNumber: 139,
                                                        columnNumber: 21
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                        onClick: (e)=>handleDownloadVideo(e, report.job_id),
                                                        disabled: downloadingVideo === report.job_id,
                                                        className: "text-gray-600 hover:text-emerald-400 p-1 rounded hover:bg-emerald-500/10 transition-colors disabled:opacity-50",
                                                        title: "Download Tactical Video",
                                                        children: downloadingVideo === report.job_id ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2d$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader2$3e$__["Loader2"], {
                                                            size: 14,
                                                            className: "animate-spin"
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                                            lineNumber: 150,
                                                            columnNumber: 25
                                                        }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$film$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Film$3e$__["Film"], {
                                                            size: 14
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                                            lineNumber: 152,
                                                            columnNumber: 25
                                                        }, this)
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                                        lineNumber: 143,
                                                        columnNumber: 21
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                        onClick: (e)=>handleDelete(e, report.id),
                                                        className: "text-gray-600 hover:text-red-500 p-1 rounded hover:bg-red-500/10 transition-colors",
                                                        title: "Delete Report",
                                                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$trash$2d$2$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Trash2$3e$__["Trash2"], {
                                                            size: 14
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                                            lineNumber: 161,
                                                            columnNumber: 23
                                                        }, this)
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                                        lineNumber: 156,
                                                        columnNumber: 21
                                                    }, this)
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                                lineNumber: 138,
                                                columnNumber: 19
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                        lineNumber: 134,
                                        columnNumber: 17
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h3", {
                                        className: "font-bold text-gray-300 text-sm mb-1 truncate",
                                        title: report.video_title,
                                        children: report.video_title
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                        lineNumber: 166,
                                        columnNumber: 17
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "text-[10px] font-mono text-gray-500 mb-4 uppercase tracking-widest",
                                        children: [
                                            report.quality_profile,
                                            " • ",
                                            report.flaw_count,
                                            " Insights"
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                        lineNumber: 169,
                                        columnNumber: 17
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "flex items-center gap-4 mb-4",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: "flex-1",
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                                        className: "text-[8px] text-gray-600 uppercase mb-1",
                                                        children: "Win Prob (Red)"
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                                        lineNumber: 175,
                                                        columnNumber: 23
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "h-1 bg-gray-900 rounded-full overflow-hidden",
                                                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                            className: "h-full bg-emerald-500",
                                                            style: {
                                                                width: `${report.win_probability?.team_red || 50}%`
                                                            }
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                                            lineNumber: 177,
                                                            columnNumber: 26
                                                        }, this)
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                                        lineNumber: 176,
                                                        columnNumber: 23
                                                    }, this)
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                                lineNumber: 174,
                                                columnNumber: 20
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: "flex-1",
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                                        className: "text-[8px] text-gray-600 uppercase mb-1",
                                                        children: "TPI (Red)"
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                                        lineNumber: 181,
                                                        columnNumber: 23
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                                        className: "text-xs font-mono text-emerald-400 font-bold",
                                                        children: Math.round(report.tactical_power?.red || 0)
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                                        lineNumber: 182,
                                                        columnNumber: 23
                                                    }, this)
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                                lineNumber: 180,
                                                columnNumber: 20
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                        lineNumber: 173,
                                        columnNumber: 17
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                lineNumber: 133,
                                columnNumber: 15
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "mt-auto flex items-center justify-between border-t border-gray-900 pt-4",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "text-[10px] font-bold text-emerald-500 uppercase tracking-widest",
                                        children: "View Analysis"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                        lineNumber: 188,
                                        columnNumber: 18
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$arrow$2d$right$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__ArrowRight$3e$__["ArrowRight"], {
                                        size: 16,
                                        className: "text-gray-600 group-hover:text-emerald-400 transition-colors transform group-hover:translate-x-1"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                        lineNumber: 189,
                                        columnNumber: 18
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                                lineNumber: 187,
                                columnNumber: 15
                            }, this)
                        ]
                    }, report.id, true, {
                        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                        lineNumber: 128,
                        columnNumber: 13
                    }, this))
            }, void 0, false, {
                fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
                lineNumber: 126,
                columnNumber: 9
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/app/workspace/components/ReportsArchive.tsx",
        lineNumber: 93,
        columnNumber: 5
    }, this);
}
_s(ReportsArchive, "oevr/tsSgJGMS7KGztnEZ0NnGDE=");
_c = ReportsArchive;
var _c;
__turbopack_context__.k.register(_c, "ReportsArchive");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/app/workspace/components/DictionaryTab.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "DictionaryTab",
    ()=>DictionaryTab
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$book$2d$open$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__BookOpen$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/book-open.mjs [app-client] (ecmascript) <export default as BookOpen>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$check$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Check$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/check.mjs [app-client] (ecmascript) <export default as Check>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$refresh$2d$cw$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__RefreshCw$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/refresh-cw.mjs [app-client] (ecmascript) <export default as RefreshCw>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$circle$2d$alert$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__AlertCircle$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/circle-alert.mjs [app-client] (ecmascript) <export default as AlertCircle>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$info$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Info$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/info.mjs [app-client] (ecmascript) <export default as Info>");
;
var _s = __turbopack_context__.k.signature();
"use client";
;
;
const DEFAULT_ROWS = [
    {
        key: "team_0",
        type: "Team",
        label: "Team A (Red)",
        defaultAlt: "Red Team"
    },
    {
        key: "team_1",
        type: "Team",
        label: "Team B (Blue)",
        defaultAlt: "Blue Team"
    },
    {
        key: "Possession",
        type: "Category",
        label: "Possession",
        defaultAlt: "Possession"
    },
    {
        key: "No Possession",
        type: "Category",
        label: "No Possession",
        defaultAlt: "No Possession"
    },
    {
        key: "Goal",
        type: "Event",
        label: "Goal",
        defaultAlt: "Goal"
    },
    {
        key: "Shot",
        type: "Event",
        label: "Shot",
        defaultAlt: "Shot"
    },
    {
        key: "Defense",
        type: "Category",
        label: "Defense",
        defaultAlt: "Defense"
    },
    {
        key: "Recovery",
        type: "Event",
        label: "Recovery",
        defaultAlt: "Recovery"
    },
    {
        key: "Attack",
        type: "Category",
        label: "Attack",
        defaultAlt: "Attack"
    },
    {
        key: "P1",
        type: "Player",
        label: "Player 1",
        defaultAlt: "Player 1"
    },
    {
        key: "P2",
        type: "Player",
        label: "Player 2",
        defaultAlt: "Player 2"
    },
    {
        key: "P3",
        type: "Player",
        label: "Player 3",
        defaultAlt: "Player 3"
    },
    {
        key: "P4",
        type: "Player",
        label: "Player 4",
        defaultAlt: "Player 4"
    },
    {
        key: "P5",
        type: "Player",
        label: "Player 5",
        defaultAlt: "Player 5"
    },
    {
        key: "P6",
        type: "Player",
        label: "Player 6",
        defaultAlt: "Player 6"
    },
    {
        key: "P7",
        type: "Player",
        label: "Player 7",
        defaultAlt: "Player 7"
    },
    {
        key: "P8",
        type: "Player",
        label: "Player 8",
        defaultAlt: "Player 8"
    },
    {
        key: "P9",
        type: "Player",
        label: "Player 9",
        defaultAlt: "Player 9"
    },
    {
        key: "P10",
        type: "Player",
        label: "Player 10",
        defaultAlt: "Player 10"
    },
    {
        key: "P11",
        type: "Player",
        label: "Player 11",
        defaultAlt: "Player 11"
    },
    {
        key: "P12",
        type: "Player",
        label: "Player 12",
        defaultAlt: "Player 12"
    },
    {
        key: "P13",
        type: "Player",
        label: "Player 13",
        defaultAlt: "Player 13"
    },
    {
        key: "P14",
        type: "Player",
        label: "Player 14",
        defaultAlt: "Player 14"
    },
    {
        key: "P15",
        type: "Player",
        label: "Player 15",
        defaultAlt: "Player 15"
    },
    {
        key: "P16",
        type: "Player",
        label: "Player 16",
        defaultAlt: "Player 16"
    },
    {
        key: "P17",
        type: "Player",
        label: "Player 17",
        defaultAlt: "Player 17"
    },
    {
        key: "P18",
        type: "Player",
        label: "Player 18",
        defaultAlt: "Player 18"
    },
    {
        key: "P19",
        type: "Player",
        label: "Player 19",
        defaultAlt: "Player 19"
    },
    {
        key: "P20",
        type: "Player",
        label: "Player 20",
        defaultAlt: "Player 20"
    }
];
function DictionaryTab({ dictionary, useAltNames, onUpdateDictionary, onToggleAltNames }) {
    _s();
    const [localDict, setLocalDict] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])({});
    const [searchTerm, setSearchTerm] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])('');
    const [filterType, setFilterType] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])('all');
    const [saveSuccess, setSaveSuccess] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(false);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "DictionaryTab.useEffect": ()=>{
            // Populate local dict state with existing dictionary values, fallback to defaultAlt
            const initial = {};
            DEFAULT_ROWS.forEach({
                "DictionaryTab.useEffect": (row)=>{
                    initial[row.key] = dictionary[row.key] !== undefined ? dictionary[row.key] : row.defaultAlt;
                }
            }["DictionaryTab.useEffect"]);
            setLocalDict(initial);
        }
    }["DictionaryTab.useEffect"], [
        dictionary
    ]);
    const handleInputChange = (key, val)=>{
        setLocalDict((prev)=>({
                ...prev,
                [key]: val
            }));
    };
    const handleSave = ()=>{
        onUpdateDictionary(localDict);
        setSaveSuccess(true);
        setTimeout(()=>setSaveSuccess(false), 2000);
    };
    const handleReset = ()=>{
        if (window.confirm("Are you sure you want to reset all alt names to default settings?")) {
            const resetDict = {};
            DEFAULT_ROWS.forEach((row)=>{
                resetDict[row.key] = row.defaultAlt;
            });
            setLocalDict(resetDict);
            onUpdateDictionary(resetDict);
        }
    };
    const filteredRows = DEFAULT_ROWS.filter((row)=>{
        const matchesSearch = row.label.toLowerCase().includes(searchTerm.toLowerCase()) || row.key.toLowerCase().includes(searchTerm.toLowerCase()) || (localDict[row.key] || '').toLowerCase().includes(searchTerm.toLowerCase());
        const matchesFilter = filterType === 'all' || row.type === filterType;
        return matchesSearch && matchesFilter;
    });
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "h-full w-full bg-[#0a0f0a] flex flex-col animate-fade-in font-sans p-8 overflow-y-auto custom-scrollbar",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "flex items-center justify-between mb-8 border-b border-gray-900 pb-6",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h1", {
                                className: "text-2xl font-bold text-gray-200 tracking-tight flex items-center gap-3",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$book$2d$open$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__BookOpen$3e$__["BookOpen"], {
                                        className: "text-emerald-500"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                        lineNumber: 102,
                                        columnNumber: 13
                                    }, this),
                                    "Category / Descriptor Names Dictionary"
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                lineNumber: 101,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                className: "text-sm text-gray-500 mt-2 font-mono",
                                children: "Map telemetry codes and player tracking tags to custom display names for reports, timeline, HUD, and AI analysis."
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                lineNumber: 105,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                        lineNumber: 100,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex items-center gap-4 bg-[#111a12] border border-emerald-500/20 px-5 py-3 rounded-xl shadow-lg",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                className: "text-xs font-mono text-gray-400",
                                children: "Alt Names Mode"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                lineNumber: 111,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                onClick: ()=>onToggleAltNames(!useAltNames),
                                className: `w-12 h-6 flex items-center rounded-full p-1 cursor-pointer transition-all duration-300 ${useAltNames ? 'bg-emerald-500' : 'bg-gray-800'}`,
                                "aria-label": "Toggle Alt Names",
                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: `bg-black w-4 h-4 rounded-full shadow-md transform transition-transform duration-300 ${useAltNames ? 'translate-x-6' : 'translate-x-0'}`
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                    lineNumber: 119,
                                    columnNumber: 13
                                }, this)
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                lineNumber: 112,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                        lineNumber: 110,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                lineNumber: 99,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "grid grid-cols-1 lg:grid-cols-[1fr_3fr] gap-8 items-start",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex flex-col gap-6",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "bg-[#111a12]/30 border border-gray-900 rounded-xl p-5 shadow-md space-y-4",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                                        className: "text-xs font-bold text-gray-400 font-mono uppercase tracking-widest flex items-center gap-2",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$info$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Info$3e$__["Info"], {
                                                size: 14,
                                                className: "text-emerald-500"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                lineNumber: 133,
                                                columnNumber: 15
                                            }, this),
                                            " System Guide"
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                        lineNumber: 132,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                        className: "text-xs text-gray-400 leading-relaxed font-sans",
                                        children: [
                                            "Remapped names are used contextually. For instance, when ",
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("strong", {
                                                className: "text-emerald-400",
                                                children: "Marcus Rashford"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                lineNumber: 136,
                                                columnNumber: 72
                                            }, this),
                                            " is assigned to ",
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("strong", {
                                                className: "text-gray-300",
                                                children: "P5"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                lineNumber: 136,
                                                columnNumber: 149
                                            }, this),
                                            ":"
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                        lineNumber: 135,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("ul", {
                                        className: "text-xs text-gray-500 space-y-2 font-mono pl-4 list-disc",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("li", {
                                                children: [
                                                    "Head badge HUD renders ",
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                        className: "text-emerald-400/90",
                                                        children: "Marcus Rashford"
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                        lineNumber: 139,
                                                        columnNumber: 42
                                                    }, this)
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                lineNumber: 139,
                                                columnNumber: 15
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("li", {
                                                children: "Chat foot-notes map P5 requests correctly"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                lineNumber: 140,
                                                columnNumber: 15
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("li", {
                                                children: "Player statistics display alt names"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                lineNumber: 141,
                                                columnNumber: 15
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                        lineNumber: 138,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                lineNumber: 131,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "bg-[#111a12]/30 border border-gray-900 rounded-xl p-5 shadow-md space-y-4",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "text-xs font-bold text-gray-400 font-mono uppercase tracking-widest",
                                        children: "Controls"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                        lineNumber: 146,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "flex flex-col gap-2",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                onClick: handleSave,
                                                className: "w-full bg-emerald-600 hover:bg-emerald-500 text-black font-bold py-2.5 px-4 rounded-lg text-sm transition-all flex items-center justify-center gap-2 shadow-lg shadow-emerald-950/20",
                                                children: saveSuccess ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Fragment"], {
                                                    children: [
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$check$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Check$3e$__["Check"], {
                                                            size: 16
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                            lineNumber: 154,
                                                            columnNumber: 21
                                                        }, this),
                                                        " Saved Successfully"
                                                    ]
                                                }, void 0, true) : "Apply Changes"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                lineNumber: 148,
                                                columnNumber: 15
                                            }, this),
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                onClick: handleReset,
                                                className: "w-full bg-transparent border border-gray-800 hover:border-red-500/50 hover:bg-red-500/5 text-gray-400 hover:text-red-400 font-bold py-2 px-4 rounded-lg text-xs transition-all flex items-center justify-center gap-2",
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$refresh$2d$cw$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__RefreshCw$3e$__["RefreshCw"], {
                                                        size: 12
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                        lineNumber: 164,
                                                        columnNumber: 17
                                                    }, this),
                                                    " Reset to Defaults"
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                lineNumber: 160,
                                                columnNumber: 15
                                            }, this)
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                        lineNumber: 147,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                lineNumber: 145,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                        lineNumber: 130,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "bg-[#111a12]/20 border border-gray-900 rounded-2xl p-6 shadow-2xl flex flex-col gap-6",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex flex-col md:flex-row gap-4 items-center justify-between",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: "flex flex-wrap gap-1 bg-[#0a0f0a] border border-gray-900 rounded-lg p-1 w-full md:w-auto",
                                        children: [
                                            'all',
                                            'Team',
                                            'Category',
                                            'Event',
                                            'Player'
                                        ].map((type)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                onClick: ()=>setFilterType(type),
                                                className: `px-3 py-1.5 rounded-md text-[10px] font-bold font-mono uppercase transition-all ${filterType === type ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'text-gray-500 hover:text-gray-300'}`,
                                                children: type
                                            }, type, false, {
                                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                lineNumber: 176,
                                                columnNumber: 17
                                            }, this))
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                        lineNumber: 174,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                        type: "text",
                                        placeholder: "Filter by name or code...",
                                        value: searchTerm,
                                        onChange: (e)=>setSearchTerm(e.target.value),
                                        className: "w-full md:w-64 bg-[#0a0f0a] border border-gray-900 rounded-lg px-4 py-2 text-xs text-gray-300 focus:outline-none focus:border-emerald-500 transition-colors"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                        lineNumber: 190,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                lineNumber: 173,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "overflow-x-auto rounded-xl border border-gray-900 bg-[#0a0f0a]/50",
                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("table", {
                                    className: "w-full text-left border-collapse text-xs",
                                    children: [
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("thead", {
                                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("tr", {
                                                className: "border-b border-gray-900 bg-[#0c120d]/60 font-mono text-gray-500 uppercase tracking-wider text-[10px]",
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("th", {
                                                        className: "py-3.5 px-4 font-bold",
                                                        children: "Descriptor Type"
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                        lineNumber: 204,
                                                        columnNumber: 19
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("th", {
                                                        className: "py-3.5 px-4 font-bold",
                                                        children: "Internal Tag / Code"
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                        lineNumber: 205,
                                                        columnNumber: 19
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("th", {
                                                        className: "py-3.5 px-4 font-bold",
                                                        children: "Default Display Name"
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                        lineNumber: 206,
                                                        columnNumber: 19
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("th", {
                                                        className: "py-3.5 px-4 font-bold w-1/3",
                                                        children: "Remapped Display Name (Alt Name)"
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                        lineNumber: 207,
                                                        columnNumber: 19
                                                    }, this)
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                lineNumber: 203,
                                                columnNumber: 17
                                            }, this)
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                            lineNumber: 202,
                                            columnNumber: 15
                                        }, this),
                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("tbody", {
                                            className: "divide-y divide-gray-900/40 text-gray-300 font-sans",
                                            children: filteredRows.length === 0 ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("tr", {
                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("td", {
                                                    colSpan: 4,
                                                    className: "py-8 text-center text-gray-600 font-mono italic",
                                                    children: "No matching mappings found."
                                                }, void 0, false, {
                                                    fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                    lineNumber: 213,
                                                    columnNumber: 21
                                                }, this)
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                lineNumber: 212,
                                                columnNumber: 19
                                            }, this) : filteredRows.map((row)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("tr", {
                                                    className: "hover:bg-emerald-500/[0.02] transition-colors",
                                                    children: [
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("td", {
                                                            className: "py-3 px-4",
                                                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                className: `px-2 py-0.5 rounded text-[9px] font-bold font-mono uppercase ${row.type === 'Team' ? 'bg-blue-500/10 text-blue-400 border border-blue-500/10' : row.type === 'Category' ? 'bg-amber-500/10 text-amber-400 border border-amber-500/10' : row.type === 'Event' ? 'bg-purple-500/10 text-purple-400 border border-purple-500/10' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/10'}`,
                                                                children: row.type
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                                lineNumber: 224,
                                                                columnNumber: 25
                                                            }, this)
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                            lineNumber: 223,
                                                            columnNumber: 23
                                                        }, this),
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("td", {
                                                            className: "py-3 px-4 font-mono text-gray-500",
                                                            children: row.key
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                            lineNumber: 233,
                                                            columnNumber: 23
                                                        }, this),
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("td", {
                                                            className: "py-3 px-4 font-medium",
                                                            children: row.label
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                            lineNumber: 234,
                                                            columnNumber: 23
                                                        }, this),
                                                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("td", {
                                                            className: "py-2 px-4",
                                                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                                                                type: "text",
                                                                value: localDict[row.key] !== undefined ? localDict[row.key] : "",
                                                                onChange: (e)=>handleInputChange(row.key, e.target.value),
                                                                placeholder: row.defaultAlt,
                                                                className: "w-full bg-[#111a12] border border-gray-900 hover:border-gray-800 focus:border-emerald-500/50 rounded-md px-3 py-1.5 text-xs text-gray-200 outline-none transition-all"
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                                lineNumber: 236,
                                                                columnNumber: 25
                                                            }, this)
                                                        }, void 0, false, {
                                                            fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                            lineNumber: 235,
                                                            columnNumber: 23
                                                        }, this)
                                                    ]
                                                }, row.key, true, {
                                                    fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                                    lineNumber: 219,
                                                    columnNumber: 21
                                                }, this))
                                        }, void 0, false, {
                                            fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                            lineNumber: 210,
                                            columnNumber: 15
                                        }, this)
                                    ]
                                }, void 0, true, {
                                    fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                    lineNumber: 201,
                                    columnNumber: 13
                                }, this)
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                lineNumber: 200,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex items-center gap-2 text-[10px] text-gray-600 font-mono",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$circle$2d$alert$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__AlertCircle$3e$__["AlertCircle"], {
                                        size: 12,
                                        className: "text-gray-700"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                        lineNumber: 252,
                                        columnNumber: 13
                                    }, this),
                                    'Remember to click "Apply Changes" to compile alterations.'
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                                lineNumber: 251,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                        lineNumber: 171,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
                lineNumber: 128,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/app/workspace/components/DictionaryTab.tsx",
        lineNumber: 97,
        columnNumber: 5
    }, this);
}
_s(DictionaryTab, "0oeEqrc3z/MArYByacIkaJakE+I=");
_c = DictionaryTab;
var _c;
__turbopack_context__.k.register(_c, "DictionaryTab");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/app/workspace/components/PlayerReports.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "PlayerReports",
    ()=>PlayerReports
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$users$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Users$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/users.mjs [app-client] (ecmascript) <export default as Users>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$user$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__User$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/user.mjs [app-client] (ecmascript) <export default as User>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$play$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Play$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/play.mjs [app-client] (ecmascript) <export default as Play>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$bot$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Bot$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/bot.mjs [app-client] (ecmascript) <export default as Bot>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$triangle$2d$alert$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__AlertTriangle$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/triangle-alert.mjs [app-client] (ecmascript) <export default as AlertTriangle>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2d$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader2$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/loader-circle.mjs [app-client] (ecmascript) <export default as Loader2>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$chart$2d$no$2d$axes$2d$column$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__BarChart2$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/chart-no-axes-column.mjs [app-client] (ecmascript) <export default as BarChart2>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$shield$2d$alert$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__ShieldAlert$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/shield-alert.mjs [app-client] (ecmascript) <export default as ShieldAlert>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$clock$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Clock$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/clock.mjs [app-client] (ecmascript) <export default as Clock>");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$film$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Film$3e$__ = __turbopack_context__.i("[project]/node_modules/lucide-react/dist/esm/icons/film.mjs [app-client] (ecmascript) <export default as Film>");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$jobs$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/api/jobs.ts [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/apiBase.ts [app-client] (ecmascript)");
;
var _s = __turbopack_context__.k.signature();
"use client";
;
;
;
;
const EVENT_NAME_LOOKUP = {
    "THR_001": "Dangerous Run",
    "THR_002": "Final-Third Entry",
    "THR_003": "Box Entry",
    "THR_004": "Transition Involvement",
    "THR_005": "Dangerous Reception",
    "THR_006": "Channel Exploitation",
    "THR_007": "Isolated Defender Exploit"
};
function PlayerReports({ job, useAltNames, dictionary, onPlayClip, onAskAI }) {
    _s();
    const [loading, setLoading] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(true);
    const [error, setError] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(null);
    const [data, setData] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(null);
    const [expandedPlayer, setExpandedPlayer] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(null);
    const [playerClips, setPlayerClips] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])({});
    const [loadingClips, setLoadingClips] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])({});
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "PlayerReports.useEffect": ()=>{
            if (!job?.jobId) {
                setLoading(false);
                return;
            }
            loadEventsData();
        }
    }["PlayerReports.useEffect"], [
        job
    ]);
    async function loadEventsData() {
        try {
            setLoading(true);
            setError(null);
            const res = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$jobs$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getJobEvents"])(job.jobId);
            setData(res);
        } catch (err) {
            console.error(err);
            setError("Failed to load player reports. Ensure the analysis is complete.");
        } finally{
            setLoading(false);
        }
    }
    const getTeamLabel = (teamId)=>{
        if (useAltNames && dictionary[teamId]) return dictionary[teamId];
        return teamId === "team_0" ? "Red Team" : "Blue Team";
    };
    const getPlayerLabel = (playerId)=>{
        const key = `P${playerId}`;
        if (useAltNames && dictionary[key]) return dictionary[key];
        return `Player ${playerId}`;
    };
    const fetchClipsForPlayer = async (playerId)=>{
        const key = `player-${playerId}`;
        if (playerClips[key] || loadingClips[key]) return;
        setLoadingClips((prev)=>({
                ...prev,
                [key]: true
            }));
        try {
            const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
            const response = await fetch(`${base}/api/v1/chat`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    ...(0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
                },
                body: JSON.stringify({
                    message: `show the clips for player ${playerId}`,
                    job_id: job.jobId,
                    llm_engine: "local"
                })
            });
            if (response.ok) {
                const chatData = await response.json();
                const clips = chatData.evidence?.clips || [];
                setPlayerClips((prev)=>({
                        ...prev,
                        [key]: clips
                    }));
            }
        } catch (err) {
            console.error("Failed to load clips for player:", err);
        } finally{
            setLoadingClips((prev)=>({
                    ...prev,
                    [key]: false
                }));
        }
    };
    const handleExpandToggle = (playerKey, playerId)=>{
        if (expandedPlayer === playerKey) {
            setExpandedPlayer(null);
        } else {
            setExpandedPlayer(playerKey);
            fetchClipsForPlayer(playerId);
        }
    };
    const threatColor = (score)=>{
        if (score >= 70) return {
            bar: "bg-red-500",
            text: "text-red-400 font-bold",
            bg: "bg-red-500/10 border-red-500/20"
        };
        if (score >= 40) return {
            bar: "bg-amber-500",
            text: "text-amber-400 font-bold",
            bg: "bg-amber-500/10 border-amber-500/20"
        };
        return {
            bar: "bg-emerald-500",
            text: "text-emerald-400 font-medium",
            bg: "bg-emerald-500/10 border-emerald-500/20"
        };
    };
    const sortedProfiles = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useMemo"])({
        "PlayerReports.useMemo[sortedProfiles]": ()=>{
            if (!data?.threat_profiles) return {
                team0: [],
                team1: []
            };
            const team0 = data.threat_profiles.filter({
                "PlayerReports.useMemo[sortedProfiles].team0": (p)=>p.team_id === 'team_0'
            }["PlayerReports.useMemo[sortedProfiles].team0"]);
            const team1 = data.threat_profiles.filter({
                "PlayerReports.useMemo[sortedProfiles].team1": (p)=>p.team_id === 'team_1'
            }["PlayerReports.useMemo[sortedProfiles].team1"]);
            return {
                team0: [
                    ...team0
                ].sort({
                    "PlayerReports.useMemo[sortedProfiles]": (a, b)=>a.threat_rank - b.threat_rank
                }["PlayerReports.useMemo[sortedProfiles]"]),
                team1: [
                    ...team1
                ].sort({
                    "PlayerReports.useMemo[sortedProfiles]": (a, b)=>a.threat_rank - b.threat_rank
                }["PlayerReports.useMemo[sortedProfiles]"])
            };
        }
    }["PlayerReports.useMemo[sortedProfiles]"], [
        data
    ]);
    if (!job) {
        return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
            className: "h-full w-full bg-[#0a0f0a] flex flex-col items-center justify-center p-8 text-center",
            children: [
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$users$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Users$3e$__["Users"], {
                    className: "text-gray-800 mb-4 animate-pulse",
                    size: 64
                }, void 0, false, {
                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                    lineNumber: 132,
                    columnNumber: 9
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                    className: "text-lg font-bold text-gray-400 font-mono uppercase tracking-widest",
                    children: "No Active Match loaded"
                }, void 0, false, {
                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                    lineNumber: 133,
                    columnNumber: 9
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                    className: "text-sm text-gray-600 max-w-sm mt-2 font-mono",
                    children: "Select a completed analysis job or upload raw footage in the dashboard to generate player reports."
                }, void 0, false, {
                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                    lineNumber: 134,
                    columnNumber: 9
                }, this)
            ]
        }, void 0, true, {
            fileName: "[project]/app/workspace/components/PlayerReports.tsx",
            lineNumber: 131,
            columnNumber: 7
        }, this);
    }
    if (loading) {
        return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
            className: "h-full w-full bg-[#0a0f0a] flex flex-col items-center justify-center gap-4",
            children: [
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2d$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader2$3e$__["Loader2"], {
                    className: "animate-spin text-emerald-500",
                    size: 48
                }, void 0, false, {
                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                    lineNumber: 142,
                    columnNumber: 9
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                    className: "text-sm font-mono text-gray-500 uppercase tracking-widest animate-pulse",
                    children: "Compiling Tactical Threat Matrices..."
                }, void 0, false, {
                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                    lineNumber: 143,
                    columnNumber: 9
                }, this)
            ]
        }, void 0, true, {
            fileName: "[project]/app/workspace/components/PlayerReports.tsx",
            lineNumber: 141,
            columnNumber: 7
        }, this);
    }
    if (error) {
        return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
            className: "h-full w-full bg-[#0a0f0a] flex flex-col items-center justify-center p-8 text-center",
            children: [
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$triangle$2d$alert$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__AlertTriangle$3e$__["AlertTriangle"], {
                    className: "text-amber-500/70 mb-4",
                    size: 48
                }, void 0, false, {
                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                    lineNumber: 151,
                    columnNumber: 9
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                    className: "text-sm font-bold text-amber-500 font-mono uppercase tracking-widest",
                    children: "Analysis telemetry Unavailable"
                }, void 0, false, {
                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                    lineNumber: 152,
                    columnNumber: 9
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                    className: "text-xs text-gray-500 max-w-md mt-2 font-sans",
                    children: error
                }, void 0, false, {
                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                    lineNumber: 153,
                    columnNumber: 9
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                    onClick: loadEventsData,
                    className: "mt-6 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20 font-mono text-xs font-bold py-2 px-6 rounded-lg transition-all",
                    children: "Retry Fetch"
                }, void 0, false, {
                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                    lineNumber: 154,
                    columnNumber: 9
                }, this)
            ]
        }, void 0, true, {
            fileName: "[project]/app/workspace/components/PlayerReports.tsx",
            lineNumber: 150,
            columnNumber: 7
        }, this);
    }
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "h-full w-full bg-[#0a0f0a] flex flex-col animate-fade-in font-sans p-8 overflow-y-auto custom-scrollbar",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8 border-b border-gray-900 pb-6",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h1", {
                                className: "text-2xl font-bold text-gray-200 tracking-tight flex items-center gap-3",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$users$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Users$3e$__["Users"], {
                                        className: "text-emerald-500"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                        lineNumber: 170,
                                        columnNumber: 13
                                    }, this),
                                    "Individual Player Tactical Reports"
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                lineNumber: 169,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                                className: "text-sm text-gray-500 mt-2 font-mono",
                                children: "Radar-tracked player threat metrics and automatic ontology event counts."
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                lineNumber: 173,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                        lineNumber: 168,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex items-center gap-6 bg-[#111a12]/30 border border-gray-900 px-6 py-3 rounded-xl font-mono text-xs",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex flex-col gap-0.5",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "text-gray-600 uppercase text-[9px] tracking-widest",
                                        children: "Active Job"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                        lineNumber: 179,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "text-emerald-400 font-bold",
                                        children: job?.file?.name || 'Historical Match'
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                        lineNumber: 180,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                lineNumber: 178,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "h-8 w-px bg-gray-800"
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                lineNumber: 182,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex flex-col gap-0.5",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "text-gray-600 uppercase text-[9px] tracking-widest",
                                        children: "Ranks Sourced"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                        lineNumber: 184,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "text-gray-400 font-bold",
                                        children: [
                                            data?.event_stats?.players_with_events ?? 0,
                                            " Players"
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                        lineNumber: 185,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                lineNumber: 183,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                        lineNumber: 177,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                lineNumber: 167,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "grid grid-cols-1 xl:grid-cols-2 gap-8 items-start",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "bg-[#111a12]/10 border border-gray-900/80 rounded-2xl p-6 shadow-xl flex flex-col gap-4",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex justify-between items-center border-b border-gray-900/60 pb-3",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                                        className: "text-sm font-bold text-red-400 font-mono uppercase tracking-widest flex items-center gap-2",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "h-2 w-2 rounded-full bg-red-500"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                lineNumber: 196,
                                                columnNumber: 15
                                            }, this),
                                            getTeamLabel('team_0')
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                        lineNumber: 195,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "text-[10px] font-mono text-gray-600 uppercase",
                                        children: "Opposition Attacking Threat"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                        lineNumber: 199,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                lineNumber: 194,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "space-y-4",
                                children: sortedProfiles.team0.length === 0 ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "py-12 text-center text-xs text-gray-600 font-mono italic",
                                    children: "No threat metrics logged for this team."
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                    lineNumber: 204,
                                    columnNumber: 15
                                }, this) : sortedProfiles.team0.map((player)=>{
                                    const colors = threatColor(player.threat_score);
                                    const playerKey = `team0-${player.player_id}`;
                                    const isExpanded = expandedPlayer === playerKey;
                                    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: `border rounded-xl transition-all ${isExpanded ? 'bg-[#152018]/40 border-emerald-500/35 shadow-lg shadow-emerald-950/5' : 'bg-black/20 border-gray-900/60 hover:bg-[#111a12]/10 hover:border-gray-800'}`,
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                onClick: ()=>handleExpandToggle(playerKey, player.player_id),
                                                className: "p-5 flex flex-col md:flex-row md:items-center justify-between gap-4 cursor-pointer select-none",
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "flex items-center gap-3",
                                                        children: [
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "p-2 bg-gray-900 rounded-lg",
                                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$user$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__User$3e$__["User"], {
                                                                    size: 18,
                                                                    className: "text-gray-400"
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                    lineNumber: 226,
                                                                    columnNumber: 27
                                                                }, this)
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 225,
                                                                columnNumber: 25
                                                            }, this),
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                children: [
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                        className: "font-bold text-gray-200 text-sm",
                                                                        children: getPlayerLabel(player.player_id)
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 229,
                                                                        columnNumber: 27
                                                                    }, this),
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                        className: "text-[10px] font-mono text-gray-600 uppercase tracking-tight mt-0.5",
                                                                        children: [
                                                                            "Threat Rank: #",
                                                                            player.threat_rank
                                                                        ]
                                                                    }, void 0, true, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 230,
                                                                        columnNumber: 27
                                                                    }, this)
                                                                ]
                                                            }, void 0, true, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 228,
                                                                columnNumber: 25
                                                            }, this)
                                                        ]
                                                    }, void 0, true, {
                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                        lineNumber: 224,
                                                        columnNumber: 23
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "flex flex-col md:items-end gap-1.5 w-full md:w-48",
                                                        children: [
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "flex justify-between w-full text-[10px] font-mono",
                                                                children: [
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                        className: "text-gray-600",
                                                                        children: "Threat Score"
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 236,
                                                                        columnNumber: 27
                                                                    }, this),
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                        className: colors.text,
                                                                        children: [
                                                                            player.threat_score.toFixed(1),
                                                                            "/100"
                                                                        ]
                                                                    }, void 0, true, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 237,
                                                                        columnNumber: 27
                                                                    }, this)
                                                                ]
                                                            }, void 0, true, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 235,
                                                                columnNumber: 25
                                                            }, this),
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "w-full h-1.5 bg-gray-950 rounded-full overflow-hidden",
                                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                    className: `h-full ${colors.bar}`,
                                                                    style: {
                                                                        width: `${player.threat_score}%`
                                                                    }
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                    lineNumber: 240,
                                                                    columnNumber: 27
                                                                }, this)
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 239,
                                                                columnNumber: 25
                                                            }, this)
                                                        ]
                                                    }, void 0, true, {
                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                        lineNumber: 234,
                                                        columnNumber: 23
                                                    }, this)
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                lineNumber: 220,
                                                columnNumber: 21
                                            }, this),
                                            isExpanded && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: "px-5 pb-5 pt-3 border-t border-gray-900/60 space-y-5 animate-fade-in text-xs font-sans",
                                                children: [
                                                    player.explanation && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "space-y-1.5",
                                                        children: [
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5",
                                                                children: [
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$shield$2d$alert$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__ShieldAlert$3e$__["ShieldAlert"], {
                                                                        size: 12,
                                                                        className: "text-emerald-500"
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 252,
                                                                        columnNumber: 31
                                                                    }, this),
                                                                    "Tactical Intelligence Verdict"
                                                                ]
                                                            }, void 0, true, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 251,
                                                                columnNumber: 29
                                                            }, this),
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "bg-black/35 rounded-lg p-4 text-gray-400 font-sans leading-relaxed whitespace-pre-line border border-gray-900/40",
                                                                children: player.explanation
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 255,
                                                                columnNumber: 29
                                                            }, this)
                                                        ]
                                                    }, void 0, true, {
                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                        lineNumber: 250,
                                                        columnNumber: 27
                                                    }, this),
                                                    player.event_counts && Object.keys(player.event_counts).length > 0 && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "space-y-2",
                                                        children: [
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5",
                                                                children: [
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$chart$2d$no$2d$axes$2d$column$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__BarChart2$3e$__["BarChart2"], {
                                                                        size: 12,
                                                                        className: "text-emerald-500"
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 265,
                                                                        columnNumber: 31
                                                                    }, this),
                                                                    "Ontology Triggers Breakdown"
                                                                ]
                                                            }, void 0, true, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 264,
                                                                columnNumber: 29
                                                            }, this),
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "flex flex-wrap gap-2",
                                                                children: Object.entries(player.event_counts).map(([code, count])=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                        className: "px-3 py-1.5 bg-black/40 border border-gray-900 hover:border-emerald-500/20 rounded-lg flex items-center gap-2",
                                                                        children: [
                                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                                className: "font-mono text-[10px] text-gray-400",
                                                                                children: EVENT_NAME_LOOKUP[code] || code
                                                                            }, void 0, false, {
                                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                lineNumber: 271,
                                                                                columnNumber: 35
                                                                            }, this),
                                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                                className: "font-mono font-bold text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded text-[10px]",
                                                                                children: count
                                                                            }, void 0, false, {
                                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                lineNumber: 272,
                                                                                columnNumber: 35
                                                                            }, this)
                                                                        ]
                                                                    }, code, true, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 270,
                                                                        columnNumber: 33
                                                                    }, this))
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 268,
                                                                columnNumber: 29
                                                            }, this)
                                                        ]
                                                    }, void 0, true, {
                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                        lineNumber: 263,
                                                        columnNumber: 27
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "space-y-2 pt-2 border-t border-gray-950",
                                                        children: [
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5",
                                                                children: [
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$film$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Film$3e$__["Film"], {
                                                                        size: 12,
                                                                        className: "text-emerald-500"
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 282,
                                                                        columnNumber: 29
                                                                    }, this),
                                                                    "Visual Evidence (AI Retrieved Clips)"
                                                                ]
                                                            }, void 0, true, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 281,
                                                                columnNumber: 27
                                                            }, this),
                                                            loadingClips[playerKey] ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "flex items-center gap-2 text-gray-500 italic py-2",
                                                                children: [
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2d$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader2$3e$__["Loader2"], {
                                                                        size: 12,
                                                                        className: "animate-spin text-emerald-500"
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 287,
                                                                        columnNumber: 31
                                                                    }, this),
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                        children: "Searching event catalog..."
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 288,
                                                                        columnNumber: 31
                                                                    }, this)
                                                                ]
                                                            }, void 0, true, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 286,
                                                                columnNumber: 29
                                                            }, this) : !playerClips[playerKey] || playerClips[playerKey].length === 0 ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "text-gray-600 italic font-mono py-1",
                                                                children: "No supported moments registered in index."
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 291,
                                                                columnNumber: 29
                                                            }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "grid grid-cols-1 sm:grid-cols-2 gap-2",
                                                                children: playerClips[playerKey].map((clip, cIdx)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                                        onClick: ()=>onPlayClip(clip.start_time_s),
                                                                        className: "w-full text-left p-3 rounded-lg bg-black/40 border border-gray-900 hover:border-emerald-500/30 hover:bg-[#111a12]/20 transition-all flex items-center justify-between group",
                                                                        children: [
                                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                className: "flex-1 min-w-0 pr-2",
                                                                                children: [
                                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                        className: "text-[11px] font-bold text-gray-200 truncate group-hover:text-emerald-400 transition-colors",
                                                                                        children: clip.label || "Key Moment"
                                                                                    }, void 0, false, {
                                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                        lineNumber: 301,
                                                                                        columnNumber: 37
                                                                                    }, this),
                                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                        className: "flex items-center gap-2 mt-1.5 text-[9px] font-mono text-gray-600",
                                                                                        children: [
                                                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                                                className: "flex items-center gap-1",
                                                                                                children: [
                                                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$clock$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Clock$3e$__["Clock"], {
                                                                                                        size: 10
                                                                                                    }, void 0, false, {
                                                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                                        lineNumber: 305,
                                                                                                        columnNumber: 81
                                                                                                    }, this),
                                                                                                    " ",
                                                                                                    Math.floor(clip.start_time_s / 60),
                                                                                                    ":",
                                                                                                    Math.floor(clip.start_time_s % 60).toString().padStart(2, '0')
                                                                                                ]
                                                                                            }, void 0, true, {
                                                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                                lineNumber: 305,
                                                                                                columnNumber: 39
                                                                                            }, this),
                                                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                                                className: "bg-emerald-500/5 text-emerald-500/50 px-1 py-0.25 rounded",
                                                                                                children: [
                                                                                                    Math.round(clip.confidence_pct || clip.relevance_score * 100),
                                                                                                    "% Match"
                                                                                                ]
                                                                                            }, void 0, true, {
                                                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                                lineNumber: 306,
                                                                                                columnNumber: 39
                                                                                            }, this)
                                                                                        ]
                                                                                    }, void 0, true, {
                                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                        lineNumber: 304,
                                                                                        columnNumber: 37
                                                                                    }, this)
                                                                                ]
                                                                            }, void 0, true, {
                                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                lineNumber: 300,
                                                                                columnNumber: 35
                                                                            }, this),
                                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                className: "h-6 w-6 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-500 group-hover:bg-emerald-500/20 group-hover:scale-105 transition-all shrink-0",
                                                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$play$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Play$3e$__["Play"], {
                                                                                    size: 8,
                                                                                    fill: "currentColor"
                                                                                }, void 0, false, {
                                                                                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                    lineNumber: 310,
                                                                                    columnNumber: 37
                                                                                }, this)
                                                                            }, void 0, false, {
                                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                lineNumber: 309,
                                                                                columnNumber: 35
                                                                            }, this)
                                                                        ]
                                                                    }, cIdx, true, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 295,
                                                                        columnNumber: 33
                                                                    }, this))
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 293,
                                                                columnNumber: 29
                                                            }, this)
                                                        ]
                                                    }, void 0, true, {
                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                        lineNumber: 280,
                                                        columnNumber: 25
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "flex justify-end pt-2 border-t border-gray-950",
                                                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                            onClick: ()=>onAskAI(`Explain how we should defend against ${getPlayerLabel(player.player_id)} (P${player.player_id}) given their threat rank of #${player.threat_rank} and tactical profile.`),
                                                            className: "bg-emerald-500 text-black hover:bg-emerald-400 font-bold px-4 py-2 rounded-lg text-xs transition-all flex items-center gap-1.5",
                                                            children: [
                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$bot$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Bot$3e$__["Bot"], {
                                                                    size: 14
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                    lineNumber: 324,
                                                                    columnNumber: 29
                                                                }, this),
                                                                " Ask AI about this Player"
                                                            ]
                                                        }, void 0, true, {
                                                            fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                            lineNumber: 320,
                                                            columnNumber: 27
                                                        }, this)
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                        lineNumber: 319,
                                                        columnNumber: 25
                                                    }, this)
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                lineNumber: 247,
                                                columnNumber: 23
                                            }, this)
                                        ]
                                    }, player.player_id, true, {
                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                        lineNumber: 211,
                                        columnNumber: 19
                                    }, this);
                                })
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                lineNumber: 202,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                        lineNumber: 193,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "bg-[#111a12]/10 border border-gray-900/80 rounded-2xl p-6 shadow-xl flex flex-col gap-4",
                        children: [
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "flex justify-between items-center border-b border-gray-900/60 pb-3",
                                children: [
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("h2", {
                                        className: "text-sm font-bold text-blue-400 font-mono uppercase tracking-widest flex items-center gap-2",
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                className: "h-2 w-2 rounded-full bg-blue-500"
                                            }, void 0, false, {
                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                lineNumber: 340,
                                                columnNumber: 15
                                            }, this),
                                            getTeamLabel('team_1')
                                        ]
                                    }, void 0, true, {
                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                        lineNumber: 339,
                                        columnNumber: 13
                                    }, this),
                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                        className: "text-[10px] font-mono text-gray-600 uppercase",
                                        children: "Opposition Attacking Threat"
                                    }, void 0, false, {
                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                        lineNumber: 343,
                                        columnNumber: 13
                                    }, this)
                                ]
                            }, void 0, true, {
                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                lineNumber: 338,
                                columnNumber: 11
                            }, this),
                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                className: "space-y-4",
                                children: sortedProfiles.team1.length === 0 ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    className: "py-12 text-center text-xs text-gray-600 font-mono italic",
                                    children: "No threat metrics logged for this team."
                                }, void 0, false, {
                                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                    lineNumber: 348,
                                    columnNumber: 15
                                }, this) : sortedProfiles.team1.map((player)=>{
                                    const colors = threatColor(player.threat_score);
                                    const playerKey = `team1-${player.player_id}`;
                                    const isExpanded = expandedPlayer === playerKey;
                                    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                        className: `border rounded-xl transition-all ${isExpanded ? 'bg-[#152018]/40 border-emerald-500/35 shadow-lg shadow-emerald-950/5' : 'bg-black/20 border-gray-900/60 hover:bg-[#111a12]/10 hover:border-gray-800'}`,
                                        children: [
                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                onClick: ()=>handleExpandToggle(playerKey, player.player_id),
                                                className: "p-5 flex flex-col md:flex-row md:items-center justify-between gap-4 cursor-pointer select-none",
                                                children: [
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "flex items-center gap-3",
                                                        children: [
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "p-2 bg-gray-900 rounded-lg",
                                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$user$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__User$3e$__["User"], {
                                                                    size: 18,
                                                                    className: "text-gray-400"
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                    lineNumber: 370,
                                                                    columnNumber: 27
                                                                }, this)
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 369,
                                                                columnNumber: 25
                                                            }, this),
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                children: [
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                        className: "font-bold text-gray-200 text-sm",
                                                                        children: getPlayerLabel(player.player_id)
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 373,
                                                                        columnNumber: 27
                                                                    }, this),
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                        className: "text-[10px] font-mono text-gray-600 uppercase tracking-tight mt-0.5",
                                                                        children: [
                                                                            "Threat Rank: #",
                                                                            player.threat_rank
                                                                        ]
                                                                    }, void 0, true, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 374,
                                                                        columnNumber: 27
                                                                    }, this)
                                                                ]
                                                            }, void 0, true, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 372,
                                                                columnNumber: 25
                                                            }, this)
                                                        ]
                                                    }, void 0, true, {
                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                        lineNumber: 368,
                                                        columnNumber: 23
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "flex flex-col md:items-end gap-1.5 w-full md:w-48",
                                                        children: [
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "flex justify-between w-full text-[10px] font-mono",
                                                                children: [
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                        className: "text-gray-600",
                                                                        children: "Threat Score"
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 380,
                                                                        columnNumber: 27
                                                                    }, this),
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                        className: colors.text,
                                                                        children: [
                                                                            player.threat_score.toFixed(1),
                                                                            "/100"
                                                                        ]
                                                                    }, void 0, true, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 381,
                                                                        columnNumber: 27
                                                                    }, this)
                                                                ]
                                                            }, void 0, true, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 379,
                                                                columnNumber: 25
                                                            }, this),
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "w-full h-1.5 bg-gray-950 rounded-full overflow-hidden",
                                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                    className: `h-full ${colors.bar}`,
                                                                    style: {
                                                                        width: `${player.threat_score}%`
                                                                    }
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                    lineNumber: 384,
                                                                    columnNumber: 27
                                                                }, this)
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 383,
                                                                columnNumber: 25
                                                            }, this)
                                                        ]
                                                    }, void 0, true, {
                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                        lineNumber: 378,
                                                        columnNumber: 23
                                                    }, this)
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                lineNumber: 364,
                                                columnNumber: 21
                                            }, this),
                                            isExpanded && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                className: "px-5 pb-5 pt-3 border-t border-gray-900/60 space-y-5 animate-fade-in text-xs font-sans",
                                                children: [
                                                    player.explanation && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "space-y-1.5",
                                                        children: [
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5",
                                                                children: [
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$shield$2d$alert$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__ShieldAlert$3e$__["ShieldAlert"], {
                                                                        size: 12,
                                                                        className: "text-emerald-500"
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 396,
                                                                        columnNumber: 31
                                                                    }, this),
                                                                    "Tactical Intelligence Verdict"
                                                                ]
                                                            }, void 0, true, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 395,
                                                                columnNumber: 29
                                                            }, this),
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "bg-black/35 rounded-lg p-4 text-gray-400 font-sans leading-relaxed whitespace-pre-line border border-gray-900/40",
                                                                children: player.explanation
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 399,
                                                                columnNumber: 29
                                                            }, this)
                                                        ]
                                                    }, void 0, true, {
                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                        lineNumber: 394,
                                                        columnNumber: 27
                                                    }, this),
                                                    player.event_counts && Object.keys(player.event_counts).length > 0 && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "space-y-2",
                                                        children: [
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5",
                                                                children: [
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$chart$2d$no$2d$axes$2d$column$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__BarChart2$3e$__["BarChart2"], {
                                                                        size: 12,
                                                                        className: "text-emerald-500"
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 409,
                                                                        columnNumber: 31
                                                                    }, this),
                                                                    "Ontology Triggers Breakdown"
                                                                ]
                                                            }, void 0, true, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 408,
                                                                columnNumber: 29
                                                            }, this),
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "flex flex-wrap gap-2",
                                                                children: Object.entries(player.event_counts).map(([code, count])=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                        className: "px-3 py-1.5 bg-black/40 border border-gray-900 hover:border-emerald-500/20 rounded-lg flex items-center gap-2",
                                                                        children: [
                                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                                className: "font-mono text-[10px] text-gray-400",
                                                                                children: EVENT_NAME_LOOKUP[code] || code
                                                                            }, void 0, false, {
                                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                lineNumber: 415,
                                                                                columnNumber: 35
                                                                            }, this),
                                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                                className: "font-mono font-bold text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded text-[10px]",
                                                                                children: count
                                                                            }, void 0, false, {
                                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                lineNumber: 416,
                                                                                columnNumber: 35
                                                                            }, this)
                                                                        ]
                                                                    }, code, true, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 414,
                                                                        columnNumber: 33
                                                                    }, this))
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 412,
                                                                columnNumber: 29
                                                            }, this)
                                                        ]
                                                    }, void 0, true, {
                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                        lineNumber: 407,
                                                        columnNumber: 27
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "space-y-2 pt-2 border-t border-gray-950",
                                                        children: [
                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "text-[9px] font-bold font-mono text-gray-500 uppercase tracking-widest flex items-center gap-1.5",
                                                                children: [
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$film$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Film$3e$__["Film"], {
                                                                        size: 12,
                                                                        className: "text-emerald-500"
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 426,
                                                                        columnNumber: 29
                                                                    }, this),
                                                                    "Visual Evidence (AI Retrieved Clips)"
                                                                ]
                                                            }, void 0, true, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 425,
                                                                columnNumber: 27
                                                            }, this),
                                                            loadingClips[playerKey] ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "flex items-center gap-2 text-gray-500 italic py-2",
                                                                children: [
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$loader$2d$circle$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Loader2$3e$__["Loader2"], {
                                                                        size: 12,
                                                                        className: "animate-spin text-emerald-500"
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 431,
                                                                        columnNumber: 31
                                                                    }, this),
                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                        children: "Searching event catalog..."
                                                                    }, void 0, false, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 432,
                                                                        columnNumber: 31
                                                                    }, this)
                                                                ]
                                                            }, void 0, true, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 430,
                                                                columnNumber: 29
                                                            }, this) : !playerClips[playerKey] || playerClips[playerKey].length === 0 ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "text-gray-600 italic font-mono py-1",
                                                                children: "No supported moments registered in index."
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 435,
                                                                columnNumber: 29
                                                            }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                className: "grid grid-cols-1 sm:grid-cols-2 gap-2",
                                                                children: playerClips[playerKey].map((clip, cIdx)=>/*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                                        onClick: ()=>onPlayClip(clip.start_time_s),
                                                                        className: "w-full text-left p-3 rounded-lg bg-black/40 border border-gray-900 hover:border-emerald-500/30 hover:bg-[#111a12]/20 transition-all flex items-center justify-between group",
                                                                        children: [
                                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                className: "flex-1 min-w-0 pr-2",
                                                                                children: [
                                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                        className: "text-[11px] font-bold text-gray-200 truncate group-hover:text-emerald-400 transition-colors",
                                                                                        children: clip.label || "Key Moment"
                                                                                    }, void 0, false, {
                                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                        lineNumber: 445,
                                                                                        columnNumber: 37
                                                                                    }, this),
                                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                        className: "flex items-center gap-2 mt-1.5 text-[9px] font-mono text-gray-600",
                                                                                        children: [
                                                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                                                className: "flex items-center gap-1",
                                                                                                children: [
                                                                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$clock$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Clock$3e$__["Clock"], {
                                                                                                        size: 10
                                                                                                    }, void 0, false, {
                                                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                                        lineNumber: 449,
                                                                                                        columnNumber: 81
                                                                                                    }, this),
                                                                                                    " ",
                                                                                                    Math.floor(clip.start_time_s / 60),
                                                                                                    ":",
                                                                                                    Math.floor(clip.start_time_s % 60).toString().padStart(2, '0')
                                                                                                ]
                                                                                            }, void 0, true, {
                                                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                                lineNumber: 449,
                                                                                                columnNumber: 39
                                                                                            }, this),
                                                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("span", {
                                                                                                className: "bg-emerald-500/5 text-emerald-500/50 px-1 py-0.25 rounded",
                                                                                                children: [
                                                                                                    Math.round(clip.confidence_pct || clip.relevance_score * 100),
                                                                                                    "% Match"
                                                                                                ]
                                                                                            }, void 0, true, {
                                                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                                lineNumber: 450,
                                                                                                columnNumber: 39
                                                                                            }, this)
                                                                                        ]
                                                                                    }, void 0, true, {
                                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                        lineNumber: 448,
                                                                                        columnNumber: 37
                                                                                    }, this)
                                                                                ]
                                                                            }, void 0, true, {
                                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                lineNumber: 444,
                                                                                columnNumber: 35
                                                                            }, this),
                                                                            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                                                className: "h-6 w-6 rounded-full bg-emerald-500/10 flex items-center justify-center text-emerald-500 group-hover:bg-emerald-500/20 group-hover:scale-105 transition-all shrink-0",
                                                                                children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$play$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Play$3e$__["Play"], {
                                                                                    size: 8,
                                                                                    fill: "currentColor"
                                                                                }, void 0, false, {
                                                                                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                    lineNumber: 454,
                                                                                    columnNumber: 37
                                                                                }, this)
                                                                            }, void 0, false, {
                                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                                lineNumber: 453,
                                                                                columnNumber: 35
                                                                            }, this)
                                                                        ]
                                                                    }, cIdx, true, {
                                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                        lineNumber: 439,
                                                                        columnNumber: 33
                                                                    }, this))
                                                            }, void 0, false, {
                                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                lineNumber: 437,
                                                                columnNumber: 29
                                                            }, this)
                                                        ]
                                                    }, void 0, true, {
                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                        lineNumber: 424,
                                                        columnNumber: 25
                                                    }, this),
                                                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                                        className: "flex justify-end pt-2 border-t border-gray-950",
                                                        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                                            onClick: ()=>onAskAI(`Explain how we should defend against ${getPlayerLabel(player.player_id)} (P${player.player_id}) given their threat rank of #${player.threat_rank} and tactical profile.`),
                                                            className: "bg-emerald-500 text-black hover:bg-emerald-400 font-bold px-4 py-2 rounded-lg text-xs transition-all flex items-center gap-1.5",
                                                            children: [
                                                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$lucide$2d$react$2f$dist$2f$esm$2f$icons$2f$bot$2e$mjs__$5b$app$2d$client$5d$__$28$ecmascript$29$__$3c$export__default__as__Bot$3e$__["Bot"], {
                                                                    size: 14
                                                                }, void 0, false, {
                                                                    fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                                    lineNumber: 468,
                                                                    columnNumber: 29
                                                                }, this),
                                                                " Ask AI about this Player"
                                                            ]
                                                        }, void 0, true, {
                                                            fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                            lineNumber: 464,
                                                            columnNumber: 27
                                                        }, this)
                                                    }, void 0, false, {
                                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                        lineNumber: 463,
                                                        columnNumber: 25
                                                    }, this)
                                                ]
                                            }, void 0, true, {
                                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                                lineNumber: 391,
                                                columnNumber: 23
                                            }, this)
                                        ]
                                    }, player.player_id, true, {
                                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                        lineNumber: 355,
                                        columnNumber: 19
                                    }, this);
                                })
                            }, void 0, false, {
                                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                                lineNumber: 346,
                                columnNumber: 11
                            }, this)
                        ]
                    }, void 0, true, {
                        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                        lineNumber: 337,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/components/PlayerReports.tsx",
                lineNumber: 191,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/app/workspace/components/PlayerReports.tsx",
        lineNumber: 165,
        columnNumber: 5
    }, this);
}
_s(PlayerReports, "oB6QocOxz2oBz4kW6FySnK6KSt0=");
_c = PlayerReports;
var _c;
__turbopack_context__.k.register(_c, "PlayerReports");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/lib/api/coach.ts [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "getCoachAdvice",
    ()=>getCoachAdvice,
    "getEnrichedReport",
    ()=>getEnrichedReport,
    "postCoachChat",
    ()=>postCoachChat
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/apiBase.ts [app-client] (ecmascript)");
;
const _sleep = (ms)=>new Promise((r)=>{
        setTimeout(r, ms);
    });
// #region agent log
function _coachDebug(hypothesisId, location, message, data) {
    fetch(`${(0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])()}/ingest/b94af6c0-0f3f-4385-ab39-095f9a480704`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-Debug-Session-Id": "bb63ae",
            ...(0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
        },
        body: JSON.stringify({
            sessionId: "bb63ae",
            hypothesisId,
            location,
            message,
            data,
            timestamp: Date.now()
        })
    }).catch(()=>{});
}
// #endregion
function coachAdviceUrl(jobId, llmEngine, skipLlm) {
    const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
    return `${base}/api/v1/coach/advice?${new URLSearchParams({
        job_id: jobId,
        llm_engine: llmEngine,
        skip_llm: skipLlm ? "true" : "false"
    }).toString()}`;
}
async function _coachFetch(url) {
    try {
        return await fetch(url, {
            headers: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
        });
    } catch (e) {
        const inner = e instanceof Error ? e.message : String(e);
        // #region agent log
        _coachDebug("A", "coach.ts:_coachFetch", "fetch_rejected", {
            inner,
            urlHost: (()=>{
                try {
                    return new URL(url).host;
                } catch  {
                    return "bad-url";
                }
            })()
        });
        // #endregion
        const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
        throw new Error(`Coach request could not reach the API (${inner}). ` + `Confirm FastAPI is running at ${base}. ` + `Safari/WebKit often reports this as "Load failed" when the connection drops or times out ` + `(a long local-Ollama refresh on /coach/advice can cause that).`);
    }
}
async function getCoachAdvice(jobId, llmEngine = "local") {
    const maxAttempts = 36;
    // #region agent log
    _coachDebug("E", "coach.ts:getCoachAdvice", "enter", {
        jobIdPrefix: jobId.slice(0, 8),
        llmEngine
    });
    // #endregion
    for(let attempt = 0; attempt < maxAttempts; attempt++){
        const res = await _coachFetch(coachAdviceUrl(jobId, llmEngine, true));
        // #region agent log
        _coachDebug("D", "coach.ts:getCoachAdvice", "poll_skip_llm_response", {
            attempt,
            status: res.status,
            ok: res.ok
        });
        // #endregion
        if (res.ok) {
            let fast;
            try {
                fast = await res.json();
            } catch (parseErr) {
                // #region agent log
                _coachDebug("E", "coach.ts:getCoachAdvice", "poll_json_parse_failed", {
                    err: parseErr instanceof Error ? parseErr.message : String(parseErr)
                });
                // #endregion
                throw new Error("Coach advice response was not valid JSON (check API base URL and that FastAPI served this request).");
            }
            if (llmEngine !== "local") {
                return fast;
            }
            const items = fast.advice_items ?? [];
            const fullyHydrated = items.length > 0 && items.every((it)=>typeof it.tactical_instruction === "string" && it.tactical_instruction.trim().length > 0);
            // #region agent log
            _coachDebug("C", "coach.ts:getCoachAdvice", "enrich_gate", {
                itemCount: items.length,
                fullyHydrated,
                skipEnrichReason: items.length === 0 ? "empty_report" : fullyHydrated ? "already_hydrated" : "needs_enrich"
            });
            // #endregion
            if (items.length === 0 || fullyHydrated) {
                return fast;
            }
            try {
                // #region agent log
                _coachDebug("A", "coach.ts:getCoachAdvice", "enrich_fetch_start", {
                    adviceLen: items.length
                });
                // #endregion
                const richRes = await _coachFetch(coachAdviceUrl(jobId, "local", false));
                // #region agent log
                _coachDebug("B", "coach.ts:getCoachAdvice", "enrich_fetch_response", {
                    status: richRes.status,
                    ok: richRes.ok
                });
                // #endregion
                if (richRes.ok) {
                    try {
                        return await richRes.json();
                    } catch (parseRich) {
                        // #region agent log
                        _coachDebug("E", "coach.ts:getCoachAdvice", "enrich_json_parse_failed", {
                            err: parseRich instanceof Error ? parseRich.message : String(parseRich)
                        });
                        // #endregion
                        return fast;
                    }
                }
                if (richRes.status === 424 || richRes.status === 500) {
                    const resSkip = await _coachFetch(coachAdviceUrl(jobId, "local", true));
                    if (resSkip.ok) {
                        return await resSkip.json();
                    }
                }
                return fast;
            } catch (enrichErr) {
                // #region agent log
                _coachDebug("E", "coach.ts:getCoachAdvice", "enrich_catch_fallback_fast", {
                    err: enrichErr instanceof Error ? enrichErr.message : String(enrichErr)
                });
                // #endregion
                return fast;
            }
        }
        const text = await res.text().catch(()=>"");
        if (res.status === 425 && attempt < maxAttempts - 1) {
            await _sleep(Math.min(2000, 400 + attempt * 150));
            continue;
        }
        if (llmEngine === "local" && (res.status === 424 || res.status === 500)) {
            const resSkip = await _coachFetch(coachAdviceUrl(jobId, llmEngine, true));
            if (resSkip.ok) {
                return await resSkip.json();
            }
        }
        throw new Error(res.status === 425 ? "Coach report was not ready after waiting (HTTP 425)." : `Coach advice failed (HTTP ${res.status}): ${text.slice(0, 200)}`);
    }
    throw new Error("Coach advice not available.");
}
async function getEnrichedReport(jobId) {
    const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
    const res = await fetch(`${base}/api/v1/jobs/${jobId}/report/enriched`, {
        headers: (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
    });
    if (!res.ok) {
        throw new Error(`Enriched report failed (HTTP ${res.status})`);
    }
    return res.json();
}
async function postCoachChat(params) {
    const base = (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getApiBaseUrl"])();
    const res = await fetch(`${base}/api/v1/chat`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            ...(0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$apiBase$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getAuthHeaders"])()
        },
        body: JSON.stringify({
            message: params.message,
            job_id: params.jobId,
            llm_engine: params.llmEngine,
            history: (params.history ?? []).map((h)=>({
                    role: h.role === "ai" ? "assistant" : "user",
                    content: h.text
                }))
        })
    });
    const data = await res.json().catch(()=>({}));
    if (!res.ok) {
        const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? {}).slice(0, 400);
        throw new Error(`Chat failed (HTTP ${res.status}): ${detail}`);
    }
    return {
        reply: data.reply ?? ""
    };
}
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
"[project]/app/workspace/page.tsx [app-client] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>WorkspacePage
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/jsx-dev-runtime.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/compiled/react/index.js [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$Titlebar$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/app/workspace/components/Titlebar.tsx [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$Sidebar$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/app/workspace/components/Sidebar.tsx [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$Hopper$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/app/workspace/components/Hopper.tsx [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$TacticalDashboard$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/app/workspace/components/TacticalDashboard.tsx [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$ReportsArchive$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/app/workspace/components/ReportsArchive.tsx [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$DictionaryTab$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/app/workspace/components/DictionaryTab.tsx [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$PlayerReports$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/app/workspace/components/PlayerReports.tsx [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$jobs$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/api/jobs.ts [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$coach$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/api/coach.ts [app-client] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$debugSessionLog$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/lib/debugSessionLog.ts [app-client] (ecmascript)");
;
var _s = __turbopack_context__.k.signature();
"use client";
;
;
;
;
;
;
;
;
;
;
;
function WorkspacePage() {
    _s();
    const [ingestionComplete, setIngestionComplete] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(false);
    const [currentView, setCurrentView] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])('dashboard');
    const [activeJob, setActiveJob] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(null);
    const [coachAdvice, setCoachAdvice] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(null);
    const [coachError, setCoachError] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(null);
    const coachDelayedRefetchSent = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useRef"])(new Set());
    // Dictionary remapping states
    const [useAltNames, setUseAltNames] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(false);
    const [dictionary, setDictionary] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])({});
    const [aiPromptOverride, setAiPromptOverride] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useState"])(null);
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "WorkspacePage.useEffect": ()=>{
            if ("TURBOPACK compile-time falsy", 0) //TURBOPACK unreachable
            ;
            if (!localStorage.getItem("gaffer-engine-type")) {
                localStorage.setItem("gaffer-engine-type", "local");
            }
            if (!localStorage.getItem("gaffer-ollama-model")) {
                localStorage.setItem("gaffer-ollama-model", "llama3");
            }
            // Load dictionary states
            const dict = localStorage.getItem("gaffer-dictionary");
            if (dict) {
                try {
                    setDictionary(JSON.parse(dict));
                } catch (e) {
                    console.error("Failed to parse dictionary", e);
                }
            }
            setUseAltNames(localStorage.getItem("gaffer-use-alt-names") === "true");
        }
    }["WorkspacePage.useEffect"], []);
    const handleUpdateDictionary = (newDict)=>{
        setDictionary(newDict);
        localStorage.setItem("gaffer-dictionary", JSON.stringify(newDict));
        if ("TURBOPACK compile-time truthy", 1) {
            window.dispatchEvent(new Event("gaffer-dictionary-changed"));
        }
    };
    const handleToggleAltNames = (val)=>{
        setUseAltNames(val);
        localStorage.setItem("gaffer-use-alt-names", val ? "true" : "false");
        if ("TURBOPACK compile-time truthy", 1) {
            window.dispatchEvent(new Event("gaffer-dictionary-changed"));
        }
    };
    const handlePlayClipAndRedirect = (startTimeS)=>{
        setCurrentView('dashboard');
        // We will find a way to notify the dashboard player. Custom event works beautifully!
        setTimeout(()=>{
            if ("TURBOPACK compile-time truthy", 1) {
                const ev = new CustomEvent("gaffer-play-clip", {
                    detail: {
                        startTimeS
                    }
                });
                window.dispatchEvent(ev);
            }
        }, 100);
    };
    const handleAskAIAndRedirect = (prompt)=>{
        setAiPromptOverride(prompt);
        setCurrentView('dashboard');
    };
    const handleIngestionComplete = async (jobId, file)=>{
        try {
            const tracking = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$jobs$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getTracking"])(jobId);
            // #region agent log
            (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$debugSessionLog$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["debugSessionLog"])({
                sessionId: 'bb63ae',
                hypothesisId: 'H5',
                location: 'workspace/page.tsx:handleIngestionComplete',
                message: 'tracking ok',
                data: {
                    jobIdPrefix: jobId.slice(0, 8),
                    frameCount: Array.isArray(tracking?.frames) ? tracking.frames.length : -1
                }
            });
            // #endregion
            setActiveJob({
                jobId,
                file,
                tracking
            });
            setCoachAdvice(null);
            setCoachError(null);
            coachDelayedRefetchSent.current.clear();
            setIngestionComplete(true);
            try {
                const llmPref = ("TURBOPACK compile-time value", "object") !== 'undefined' && localStorage.getItem('gaffer-engine-type') === 'cloud' ? 'cloud' : 'local';
                const advice = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$coach$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getCoachAdvice"])(jobId, llmPref);
                try {
                    const enrichedCards = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$coach$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getEnrichedReport"])(jobId);
                    if (enrichedCards && enrichedCards.length > 0) {
                        advice.advice_items = enrichedCards;
                    }
                } catch (enrichErr) {
                    console.warn("Failed to load enriched report, using standard advice:", enrichErr);
                }
                setCoachAdvice(advice);
            } catch (ce) {
                setCoachError(ce instanceof Error ? ce.message : String(ce));
            }
        } catch (e) {
            console.error(e);
            // #region agent log
            (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$debugSessionLog$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["debugSessionLog"])({
                sessionId: 'bb63ae',
                hypothesisId: 'H1-H5',
                location: 'workspace/page.tsx:handleIngestionComplete',
                message: 'tracking fetch failed',
                data: {
                    jobIdPrefix: jobId.slice(0, 8),
                    err: String(e)
                }
            });
            // #endregion
            setActiveJob({
                jobId,
                file,
                tracking: null
            });
            setCoachAdvice(null);
            setCoachError(null);
            setIngestionComplete(true);
        }
    };
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$index$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["useEffect"])({
        "WorkspacePage.useEffect": ()=>{
            const jobId = activeJob?.jobId;
            if (!ingestionComplete || !jobId || coachError) return;
            if (coachAdvice === null) return;
            if ((coachAdvice.advice_items?.length ?? 0) > 0) return;
            if (coachDelayedRefetchSent.current.has(jobId)) return;
            let cancelled = false;
            const handle = window.setTimeout({
                "WorkspacePage.useEffect.handle": ()=>{
                    if (cancelled) return;
                    coachDelayedRefetchSent.current.add(jobId);
                    void ({
                        "WorkspacePage.useEffect.handle": async ()=>{
                            try {
                                const llmPref = ("TURBOPACK compile-time value", "object") !== 'undefined' && localStorage.getItem('gaffer-engine-type') === 'cloud' ? 'cloud' : 'local';
                                const advice = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$coach$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getCoachAdvice"])(jobId, llmPref);
                                try {
                                    const enrichedCards = await (0, __TURBOPACK__imported__module__$5b$project$5d2f$lib$2f$api$2f$coach$2e$ts__$5b$app$2d$client$5d$__$28$ecmascript$29$__["getEnrichedReport"])(jobId);
                                    if (enrichedCards && enrichedCards.length > 0) {
                                        advice.advice_items = enrichedCards;
                                    }
                                } catch (enrichErr) {
                                    console.warn("Delayed refetch failed to load enriched report:", enrichErr);
                                }
                                setCoachAdvice(advice);
                            } catch  {
                            /* ignore: primary coach path already surfaced errors */ }
                        }
                    })["WorkspacePage.useEffect.handle"]();
                }
            }["WorkspacePage.useEffect.handle"], 4000);
            return ({
                "WorkspacePage.useEffect": ()=>{
                    cancelled = true;
                    window.clearTimeout(handle);
                }
            })["WorkspacePage.useEffect"];
        }
    }["WorkspacePage.useEffect"], [
        ingestionComplete,
        activeJob?.jobId,
        coachError,
        coachAdvice,
        coachAdvice?.advice_items?.length
    ]);
    const handleOpenReport = (report)=>{
        // Transform persistent report into active job state
        setActiveJob({
            jobId: report.job_id || report.id,
            file: new File([], report.video_title || 'Historical Match'),
            tracking: report.telemetry || null,
            isHistorical: true
        });
        setCoachAdvice({
            generated_at: report.metadata?.saved_at || new Date().toISOString(),
            pipeline: report.pipeline || {},
            advice_items: report.advice_items || []
        });
        setIngestionComplete(true);
        setCurrentView('dashboard');
    };
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "h-screen w-screen overflow-hidden flex flex-col bg-[#0a0f0a] text-gray-300 antialiased selection:bg-emerald-500/30 selection:text-emerald-300",
        children: [
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$Titlebar$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Titlebar"], {}, void 0, false, {
                fileName: "[project]/app/workspace/page.tsx",
                lineNumber: 200,
                columnNumber: 7
            }, this),
            /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                className: "flex flex-1 overflow-hidden relative",
                children: [
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$Sidebar$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Sidebar"], {
                        currentView: currentView,
                        setCurrentView: setCurrentView
                    }, void 0, false, {
                        fileName: "[project]/app/workspace/page.tsx",
                        lineNumber: 202,
                        columnNumber: 9
                    }, this),
                    /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                        className: "flex-1 relative flex flex-col min-w-0 bg-[#050805]",
                        children: currentView === 'reports' ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$ReportsArchive$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["ReportsArchive"], {
                            onOpenReport: handleOpenReport
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/page.tsx",
                            lineNumber: 205,
                            columnNumber: 13
                        }, this) : currentView === 'dictionary' ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$DictionaryTab$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["DictionaryTab"], {
                            dictionary: dictionary,
                            useAltNames: useAltNames,
                            onUpdateDictionary: handleUpdateDictionary,
                            onToggleAltNames: handleToggleAltNames
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/page.tsx",
                            lineNumber: 207,
                            columnNumber: 13
                        }, this) : currentView === 'players' ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$PlayerReports$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["PlayerReports"], {
                            job: activeJob,
                            useAltNames: useAltNames,
                            dictionary: dictionary,
                            onPlayClip: handlePlayClipAndRedirect,
                            onAskAI: handleAskAIAndRedirect
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/page.tsx",
                            lineNumber: 214,
                            columnNumber: 13
                        }, this) : !ingestionComplete ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$Hopper$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["Hopper"], {
                            onComplete: handleIngestionComplete
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/page.tsx",
                            lineNumber: 222,
                            columnNumber: 13
                        }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$compiled$2f$react$2f$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$client$5d$__$28$ecmascript$29$__["jsxDEV"])(__TURBOPACK__imported__module__$5b$project$5d2f$app$2f$workspace$2f$components$2f$TacticalDashboard$2e$tsx__$5b$app$2d$client$5d$__$28$ecmascript$29$__["default"], {
                            job: activeJob,
                            coachAdvice: coachAdvice,
                            coachError: coachError,
                            useAltNames: useAltNames,
                            dictionary: dictionary,
                            aiPromptOverride: aiPromptOverride,
                            setAiPromptOverride: setAiPromptOverride
                        }, void 0, false, {
                            fileName: "[project]/app/workspace/page.tsx",
                            lineNumber: 224,
                            columnNumber: 13
                        }, this)
                    }, void 0, false, {
                        fileName: "[project]/app/workspace/page.tsx",
                        lineNumber: 203,
                        columnNumber: 9
                    }, this)
                ]
            }, void 0, true, {
                fileName: "[project]/app/workspace/page.tsx",
                lineNumber: 201,
                columnNumber: 7
            }, this)
        ]
    }, void 0, true, {
        fileName: "[project]/app/workspace/page.tsx",
        lineNumber: 199,
        columnNumber: 5
    }, this);
}
_s(WorkspacePage, "p+7J2Rc3UXMXn0QVoLeXDVJNZ4g=");
_c = WorkspacePage;
var _c;
__turbopack_context__.k.register(_c, "WorkspacePage");
if (typeof globalThis.$RefreshHelpers$ === 'object' && globalThis.$RefreshHelpers !== null) {
    __turbopack_context__.k.registerExports(__turbopack_context__.m, globalThis.$RefreshHelpers$);
}
}),
]);

//# sourceMappingURL=_0m93qzk._.js.map