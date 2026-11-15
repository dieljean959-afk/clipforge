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

// node_modules/@radix-ui/react-separator/dist/index.mjs
var React = __toESM(require_react(), 1);
var import_jsx_runtime = __toESM(require_jsx_runtime(), 1);
var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
var Orientation = {
  Horizontal: "horizontal",
  Vertical: "vertical"
};
var DEFAULT_ORIENTATION = Orientation.Horizontal;
var ORIENTATIONS = Object.values(Orientation);
var Separator = React.forwardRef(
  __name(function Separator2(props, forwardedRef) {
    const { decorative, orientation: orientationProp = DEFAULT_ORIENTATION, ...domProps } = props;
    const orientation = isValidOrientation(orientationProp) ? orientationProp : DEFAULT_ORIENTATION;
    const ariaOrientation = orientation === Orientation.Vertical ? orientation : void 0;
    const semanticProps = decorative ? { role: "none" } : { "aria-orientation": ariaOrientation, role: "separator" };
    return (0, import_jsx_runtime.jsx)(
      Primitive.div,
      {
        "data-orientation": orientation,
        ...semanticProps,
        ...domProps,
        ref: forwardedRef
      }
    );
  }, "Separator")
);
function isValidOrientation(orientation) {
  return ORIENTATIONS.includes(orientation);
}
__name(isValidOrientation, "isValidOrientation");
export {
  Separator as Root,
  Separator
};
//# sourceMappingURL=@radix-ui_react-separator.js.map
