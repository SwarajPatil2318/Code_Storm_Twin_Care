import React from "react";
import { createRoot } from "react-dom/client";
import TwinLayout from "./platform/TwinLayout";
import "./styles.css";
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <TwinLayout />
  </React.StrictMode>,
);
