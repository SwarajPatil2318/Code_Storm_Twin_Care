import {
  createContext,
  lazy,
  Suspense,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import type { ReactNode } from "react";
import {
  Activity,
  ChartNoAxesCombined,
  FileText,
  FlaskConical,
  GitBranch,
  Heart,
  Home,
  Settings,
  ShieldCheck,
  TriangleAlert,
  UserRound,
} from "lucide-react";
import "./platform.css";
type Module = "cardiovascular" | "metabolic";
type Chrome = {
  header: ReactNode;
  onNavigate: (target: string) => void;
  view?: string;
};
const ChromeContext = createContext<(chrome: Chrome) => void>(() => {});
export function usePlatformChrome(
  active: boolean,
  chrome: Chrome,
  deps: unknown[],
) {
  const register = useContext(ChromeContext),
    ref = useRef(chrome);
  ref.current = chrome;
  useLayoutEffect(() => {
    if (active)
      register({
        ...ref.current,
        onNavigate: (target) => ref.current.onNavigate(target),
      });
  }, [active, register, ...deps]);
}
const Cardiovascular = lazy(() => import("../CardiovascularApp"));
const Metabolic = lazy(() => import("../metabolic/MetabolicApp"));
const routeModule = (): Module =>
  location.pathname.includes("metabolic") ? "metabolic" : "cardiovascular";
export default function TwinLayout() {
  const [module, setModule] = useState<Module>(routeModule),
    [visited, setVisited] = useState<Set<Module>>(
      () => new Set([routeModule()]),
    ),
    [chrome, setChrome] = useState<Chrome>({
      header: null,
      onNavigate: () => {},
    });
  const content = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches)
      content.current?.animate(
        [
          { opacity: 0.6, transform: "translateY(3px)" },
          { opacity: 1, transform: "translateY(0)" },
        ],
        { duration: 200, easing: "ease-out" },
      );
  }, [module]);
  const register = useCallback((value: Chrome) => setChrome(value), []);
  const change = (target: Module, push = true) => {
    setModule(target);
    setVisited((old) => new Set([...old, target]));
    if (push && routeModule() !== target)
      history.pushState({}, "", `/twin/${target}`);
  };
  useLayoutEffect(() => {
    const pop = () => change(routeModule(), false);
    window.addEventListener("popstate", pop);
    return () => window.removeEventListener("popstate", pop);
  }, []);
  const nav = [
    ["Home", Home],
    ["Clinical Details", UserRound],
    ["Timeline", ChartNoAxesCombined],
    ["Evidence", ShieldCheck],
    ["Architecture", GitBranch],
    ["Settings", Settings],
  ] as const;
  return (
    <ChromeContext.Provider value={register}>
      <div
        className={`twin-platform ${module === "metabolic" ? "metabolic" : "app-shell"} ${new URLSearchParams(location.search).get("finish") === "soft" ? "finish-soft" : ""}`}
      >
        <aside className="twin-rail">
          <button
            className="twin-rail-brand"
            aria-label="TwinCare home"
            onClick={() => chrome.onNavigate("Home")}
          >
            <Heart size={25} />
          </button>
          <nav aria-label="Product navigation">
            {nav.map(([name, Icon]) => (
              <button
                key={name}
                aria-label={name}
                title={name}
                aria-current={
                  (chrome.view ?? "Home") === name ? "page" : undefined
                }
                onClick={() => chrome.onNavigate(name)}
              >
                <Icon size={21} />
              </button>
            ))}
          </nav>
        </aside>
        <div className="twin-workspace">
          <header className="twin-header">
            <button
              className="twin-brand"
              onClick={() => chrome.onNavigate("Home")}
            >
              <Heart size={27} />
              <span>TwinCare</span>
            </button>
            <div
              className="twin-module-switcher"
              role="group"
              aria-label="Twin module"
            >
              {(
                [
                  ["cardiovascular", "Cardiovascular Twin", Heart],
                  ["metabolic", "Metabolic Twin", Activity],
                ] as const
              ).map(([id, label, Icon]) => (
                <button
                  key={id}
                  aria-pressed={module === id}
                  onClick={() => change(id)}
                >
                  <Icon size={16} />
                  <span>{label}</span>
                </button>
              ))}
            </div>
            <div className="twin-header-context">{chrome.header}</div>
          </header>
          <div className="twin-module-content" ref={content}>
            <Suspense
              fallback={
                <p className="twin-loading" role="status">
                  Loading TwinCare…
                </p>
              }
            >
              {visited.has("cardiovascular") && (
                <div hidden={module !== "cardiovascular"}>
                  <Cardiovascular active={module === "cardiovascular"} />
                </div>
              )}
              {visited.has("metabolic") && (
                <div hidden={module !== "metabolic"}>
                  <Metabolic active={module === "metabolic"} />
                </div>
              )}
            </Suspense>
          </div>
          <footer className="twin-evidence" aria-label="Evidence"><button onClick={() => chrome.onNavigate("Evidence")} title="Review source details and model evidence"><ShieldCheck size={14}/>Evidence &amp; source details</button></footer>
        </div>
      </div>
    </ChromeContext.Provider>
  );
}
