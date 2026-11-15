"use client";
import {
  Primitive
} from "./chunk-ZRCZ3OV4.js";
import "./chunk-IEVOZGNX.js";
import {
  require_jsx_runtime
} from "./chunk-XOQ4UB3A.js";
import "./chunk-4U7QFG22.js";
import {
  require_react
} from "./chunk-7KAG3XWH.js";
import {
  __toESM
} from "./chunk-M2PPBBZ4.js";

// node_modules/@radix-ui/react-label/dist/index.mjs
var React = __toESM(require_react(), 1);
var import_jsx_runtime = __toESM(require_jsx_runtime(), 1);
var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
var Label = React.forwardRef(
  __name(function Label2(props, forwardedRef) {
    return (0, import_jsx_runtime.jsx)(
      Primitive.label,
      {
        ...props,
        ref: forwardedRef,
        onMouseDown: (event) => {
          const target = event.target;
          if (target.closest("button, input, select, textarea")) return;
          props.onMouseDown?.(event);
          if (!event.defaultPrevented && event.detail > 1) event.preventDefault();
        }
      }
    );
  }, "Label")
);
export {
  Label,
  Label as Root
};
//# sourceMappingURL=@radix-ui_react-label.js.map
