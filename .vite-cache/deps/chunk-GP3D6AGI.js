import {
  useLayoutEffect2
} from "./chunk-T6UOJQWQ.js";
import {
  Primitive
} from "./chunk-WJBSKEQX.js";
import {
  require_jsx_runtime
} from "./chunk-BJIITAPU.js";
import {
  require_react_dom
} from "./chunk-GMNIVPK7.js";
import {
  require_react
} from "./chunk-JI3JN37G.js";
import {
  __toESM
} from "./chunk-PR4QN5HX.js";

// node_modules/@radix-ui/react-portal/dist/index.mjs
var React = __toESM(require_react(), 1);
var ReactDOM = __toESM(require_react_dom(), 1);
var import_jsx_runtime = __toESM(require_jsx_runtime(), 1);
var PORTAL_NAME = "Portal";
var Portal = React.forwardRef((props, forwardedRef) => {
  const { container: containerProp, ...portalProps } = props;
  const [mounted, setMounted] = React.useState(false);
  useLayoutEffect2(() => setMounted(true), []);
  const container = containerProp || mounted && globalThis?.document?.body;
  return container ? ReactDOM.createPortal((0, import_jsx_runtime.jsx)(Primitive.div, { ...portalProps, ref: forwardedRef }), container) : null;
});
Portal.displayName = PORTAL_NAME;

export {
  Portal
};
//# sourceMappingURL=chunk-GP3D6AGI.js.map
