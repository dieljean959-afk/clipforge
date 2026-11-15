import {
  require_jsx_runtime
} from "./chunk-XOQ4UB3A.js";
import {
  require_react
} from "./chunk-7KAG3XWH.js";
import {
  __toESM
} from "./chunk-M2PPBBZ4.js";

// node_modules/@radix-ui/react-direction/dist/index.mjs
var React = __toESM(require_react(), 1);
var import_jsx_runtime = __toESM(require_jsx_runtime(), 1);
var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
var Direction = {
  LTR: "ltr",
  RTL: "rtl"
};
var DirectionContext = React.createContext(void 0);
var DirectionProvider = __name((props) => {
  const { dir, children } = props;
  return (0, import_jsx_runtime.jsx)(DirectionContext.Provider, { value: dir, children });
}, "DirectionProvider");
function useDirection(localDir) {
  const globalDir = React.useContext(DirectionContext);
  return localDir || globalDir || Direction.LTR;
}
__name(useDirection, "useDirection");

export {
  useDirection
};
//# sourceMappingURL=chunk-EMHYOJUE.js.map
