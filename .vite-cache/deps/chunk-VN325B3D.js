import {
  useControllableState
} from "./chunk-NLDEGCMH.js";
import {
  composeEventHandlers
} from "./chunk-E24ISM2B.js";
import {
  Primitive
} from "./chunk-ZRCZ3OV4.js";
import {
  require_jsx_runtime
} from "./chunk-XOQ4UB3A.js";
import {
  require_react
} from "./chunk-7KAG3XWH.js";
import {
  __toESM
} from "./chunk-M2PPBBZ4.js";

// node_modules/@radix-ui/react-toggle/dist/index.mjs
var React = __toESM(require_react(), 1);
var import_jsx_runtime = __toESM(require_jsx_runtime(), 1);
var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
var NAME = "Toggle";
var Toggle = React.forwardRef(
  __name(function Toggle2(props, forwardedRef) {
    const { pressed: pressedProp, defaultPressed, onPressedChange, ...buttonProps } = props;
    const [pressed, setPressed] = useControllableState({
      prop: pressedProp,
      onChange: onPressedChange,
      defaultProp: defaultPressed ?? false,
      caller: NAME
    });
    return (0, import_jsx_runtime.jsx)(
      Primitive.button,
      {
        type: "button",
        "aria-pressed": pressed,
        "data-state": pressed ? "on" : "off",
        "data-disabled": props.disabled ? "" : void 0,
        ...buttonProps,
        ref: forwardedRef,
        onClick: composeEventHandlers(props.onClick, () => {
          if (!props.disabled) {
            setPressed(!pressed);
          }
        })
      }
    );
  }, "Toggle")
);

export {
  Toggle
};
//# sourceMappingURL=chunk-VN325B3D.js.map
