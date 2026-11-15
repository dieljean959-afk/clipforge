"use client";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogOverlay,
  DialogPortal,
  DialogTitle,
  DialogTrigger,
  createDialogScope
} from "./chunk-R7357JPJ.js";
import "./chunk-QFUDUECV.js";
import "./chunk-KHPBFKCS.js";
import "./chunk-ZAPJPNAP.js";
import "./chunk-X7Z4QOHD.js";
import "./chunk-ROQYZUNY.js";
import "./chunk-NLDEGCMH.js";
import "./chunk-ETDQP2KF.js";
import {
  composeEventHandlers
} from "./chunk-E24ISM2B.js";
import "./chunk-V4XET5H7.js";
import {
  createContextScope
} from "./chunk-NYJQAXGL.js";
import "./chunk-ZRCZ3OV4.js";
import {
  useComposedRefs
} from "./chunk-IEVOZGNX.js";
import "./chunk-BYNCKTIQ.js";
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

// node_modules/@radix-ui/react-alert-dialog/dist/index.mjs
var React = __toESM(require_react(), 1);
var import_jsx_runtime = __toESM(require_jsx_runtime(), 1);
var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
var ROOT_NAME = "AlertDialog";
var [createAlertDialogContext, createAlertDialogScope] = createContextScope(ROOT_NAME, [
  createDialogScope
]);
var useDialogScope = createDialogScope();
var AlertDialog = __name((props) => {
  const { __scopeAlertDialog, ...alertDialogProps } = props;
  const dialogScope = useDialogScope(__scopeAlertDialog);
  return (0, import_jsx_runtime.jsx)(Dialog, { ...dialogScope, ...alertDialogProps, modal: true });
}, "AlertDialog");
var AlertDialogTrigger = React.forwardRef(
  __name(function AlertDialogTrigger2(props, forwardedRef) {
    const { __scopeAlertDialog, ...triggerProps } = props;
    const dialogScope = useDialogScope(__scopeAlertDialog);
    return (0, import_jsx_runtime.jsx)(DialogTrigger, { ...dialogScope, ...triggerProps, ref: forwardedRef });
  }, "AlertDialogTrigger")
);
var AlertDialogPortal = __name((props) => {
  const { __scopeAlertDialog, ...portalProps } = props;
  const dialogScope = useDialogScope(__scopeAlertDialog);
  return (0, import_jsx_runtime.jsx)(DialogPortal, { ...dialogScope, ...portalProps });
}, "AlertDialogPortal");
var AlertDialogOverlay = React.forwardRef(
  __name(function AlertDialogOverlay2(props, forwardedRef) {
    const { __scopeAlertDialog, ...overlayProps } = props;
    const dialogScope = useDialogScope(__scopeAlertDialog);
    return (0, import_jsx_runtime.jsx)(DialogOverlay, { ...dialogScope, ...overlayProps, ref: forwardedRef });
  }, "AlertDialogOverlay")
);
var CONTENT_NAME = "AlertDialogContent";
var [AlertDialogContentProvider, useAlertDialogContentContext] = createAlertDialogContext(CONTENT_NAME);
var AlertDialogContent = React.forwardRef(
  __name(function AlertDialogContent2(props, forwardedRef) {
    const { __scopeAlertDialog, children, ...contentProps } = props;
    const dialogScope = useDialogScope(__scopeAlertDialog);
    const contentRef = React.useRef(null);
    const composedRefs = useComposedRefs(forwardedRef, contentRef);
    const cancelRef = React.useRef(null);
    return (0, import_jsx_runtime.jsx)(AlertDialogContentProvider, { scope: __scopeAlertDialog, cancelRef, children: (0, import_jsx_runtime.jsx)(
      DialogContent,
      {
        role: "alertdialog",
        ...dialogScope,
        ...contentProps,
        ref: composedRefs,
        onOpenAutoFocus: composeEventHandlers(contentProps.onOpenAutoFocus, (event) => {
          event.preventDefault();
          cancelRef.current?.focus({ preventScroll: true });
        }),
        onPointerDownOutside: (event) => event.preventDefault(),
        onInteractOutside: (event) => event.preventDefault(),
        children
      }
    ) });
  }, "AlertDialogContent")
);
var AlertDialogTitle = React.forwardRef(
  __name(function AlertDialogTitle2(props, forwardedRef) {
    const { __scopeAlertDialog, ...titleProps } = props;
    const dialogScope = useDialogScope(__scopeAlertDialog);
    return (0, import_jsx_runtime.jsx)(DialogTitle, { ...dialogScope, ...titleProps, ref: forwardedRef });
  }, "AlertDialogTitle")
);
var AlertDialogDescription = React.forwardRef(__name(function AlertDialogDescription2(props, forwardedRef) {
  const { __scopeAlertDialog, ...descriptionProps } = props;
  const dialogScope = useDialogScope(__scopeAlertDialog);
  return (0, import_jsx_runtime.jsx)(DialogDescription, { ...dialogScope, ...descriptionProps, ref: forwardedRef });
}, "AlertDialogDescription"));
var AlertDialogAction = React.forwardRef(
  __name(function AlertDialogAction2(props, forwardedRef) {
    const { __scopeAlertDialog, ...actionProps } = props;
    const dialogScope = useDialogScope(__scopeAlertDialog);
    return (0, import_jsx_runtime.jsx)(DialogClose, { ...dialogScope, ...actionProps, ref: forwardedRef });
  }, "AlertDialogAction")
);
var CANCEL_NAME = "AlertDialogCancel";
var AlertDialogCancel = React.forwardRef(
  __name(function AlertDialogCancel2(props, forwardedRef) {
    const { __scopeAlertDialog, ...cancelProps } = props;
    const { cancelRef } = useAlertDialogContentContext(CANCEL_NAME, __scopeAlertDialog);
    const dialogScope = useDialogScope(__scopeAlertDialog);
    const ref = useComposedRefs(forwardedRef, cancelRef);
    return (0, import_jsx_runtime.jsx)(DialogClose, { ...dialogScope, ...cancelProps, ref });
  }, "AlertDialogCancel")
);
export {
  AlertDialogAction as Action,
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogOverlay,
  AlertDialogPortal,
  AlertDialogTitle,
  AlertDialogTrigger,
  AlertDialogCancel as Cancel,
  AlertDialogContent as Content,
  AlertDialogDescription as Description,
  AlertDialogOverlay as Overlay,
  AlertDialogPortal as Portal,
  AlertDialog as Root,
  AlertDialogTitle as Title,
  AlertDialogTrigger as Trigger,
  createAlertDialogScope
};
//# sourceMappingURL=@radix-ui_react-alert-dialog.js.map
