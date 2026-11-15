import {
  useLayoutEffect2
} from "./chunk-V4XET5H7.js";
import {
  require_react
} from "./chunk-7KAG3XWH.js";
import {
  __toESM
} from "./chunk-M2PPBBZ4.js";

// node_modules/@radix-ui/react-use-callback-ref/dist/index.mjs
var React = __toESM(require_react(), 1);
var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
var useReactInsertionEffect = React[" useInsertionEffect ".trim().toString()];
function useCallbackRef(callback) {
  const callbackRef = React.useRef(callback);
  if (typeof useReactInsertionEffect === "function") {
    useReactInsertionEffect(() => {
      callbackRef.current = callback;
    });
  } else {
    useLayoutEffect2(() => {
      callbackRef.current = callback;
    });
  }
  return React.useMemo(() => ((...args) => callbackRef.current?.(...args)), []);
}
__name(useCallbackRef, "useCallbackRef");

export {
  useCallbackRef
};
//# sourceMappingURL=chunk-X7Z4QOHD.js.map
