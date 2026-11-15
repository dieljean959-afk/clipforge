"use client";
import {
  Popper,
  PopperAnchor,
  PopperArrow,
  PopperContent,
  createPopperScope
} from "./chunk-5GE3UBVK.js";
import "./chunk-O2ZAVM3M.js";
import {
  useFocusGuards
} from "./chunk-QFUDUECV.js";
import {
  FocusScope,
  FocusScopeBranchProvider,
  Portal,
  useFocusScopeBranch,
  useFocusScopeBranchRegistry
} from "./chunk-KHPBFKCS.js";
import {
  DismissableLayer
} from "./chunk-ZAPJPNAP.js";
import "./chunk-X7Z4QOHD.js";
import {
  useId
} from "./chunk-ROQYZUNY.js";
import {
  useControllableState
} from "./chunk-NLDEGCMH.js";
import {
  Presence
} from "./chunk-ETDQP2KF.js";
import {
  composeEventHandlers
} from "./chunk-E24ISM2B.js";
import {
  useLayoutEffect2
} from "./chunk-V4XET5H7.js";
import {
  createContextScope
} from "./chunk-NYJQAXGL.js";
import {
  Primitive
} from "./chunk-ZRCZ3OV4.js";
import {
  createSlot,
  useComposedRefs
} from "./chunk-IEVOZGNX.js";
import {
  Combination_default,
  hideOthers
} from "./chunk-BYNCKTIQ.js";
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

// node_modules/@radix-ui/react-popover/dist/index.mjs
var React = __toESM(require_react(), 1);
var import_jsx_runtime = __toESM(require_jsx_runtime(), 1);
var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
var POPOVER_NAME = "Popover";
var [createPopoverContext, createPopoverScope] = createContextScope(POPOVER_NAME, [
  createPopperScope
]);
var usePopperScope = createPopperScope();
var [PopoverProvider, usePopoverContext] = createPopoverContext(POPOVER_NAME);
var Popover = __name((props) => {
  const {
    __scopePopover,
    children,
    open: openProp,
    defaultOpen,
    onOpenChange,
    modal = false
  } = props;
  const popperScope = usePopperScope(__scopePopover);
  const triggerRef = React.useRef(null);
  const [hasCustomAnchor, setHasCustomAnchor] = React.useState(false);
  const [titleCount, setTitleCount] = React.useState(0);
  const [descriptionCount, setDescriptionCount] = React.useState(0);
  const [open, setOpen] = useControllableState({
    prop: openProp,
    defaultProp: defaultOpen ?? false,
    onChange: onOpenChange,
    caller: POPOVER_NAME
  });
  return (0, import_jsx_runtime.jsx)(Popper, { ...popperScope, children: (0, import_jsx_runtime.jsx)(
    PopoverProvider,
    {
      scope: __scopePopover,
      contentId: useId(),
      titleId: useId(),
      descriptionId: useId(),
      titlePresent: titleCount > 0,
      descriptionPresent: descriptionCount > 0,
      setTitleCount,
      setDescriptionCount,
      triggerRef,
      open,
      onOpenChange: setOpen,
      onOpenToggle: React.useCallback(() => setOpen((prevOpen) => !prevOpen), [setOpen]),
      hasCustomAnchor,
      onCustomAnchorAdd: React.useCallback(() => setHasCustomAnchor(true), []),
      onCustomAnchorRemove: React.useCallback(() => setHasCustomAnchor(false), []),
      modal,
      children
    }
  ) });
}, "Popover");
var ANCHOR_NAME = "PopoverAnchor";
var PopoverAnchor = React.forwardRef(
  __name(function PopoverAnchor2(props, forwardedRef) {
    const { __scopePopover, ...anchorProps } = props;
    const context = usePopoverContext(ANCHOR_NAME, __scopePopover);
    const popperScope = usePopperScope(__scopePopover);
    const { onCustomAnchorAdd, onCustomAnchorRemove } = context;
    React.useEffect(() => {
      onCustomAnchorAdd();
      return () => onCustomAnchorRemove();
    }, [onCustomAnchorAdd, onCustomAnchorRemove]);
    return (0, import_jsx_runtime.jsx)(PopperAnchor, { ...popperScope, ...anchorProps, ref: forwardedRef });
  }, "PopoverAnchor")
);
var TRIGGER_NAME = "PopoverTrigger";
var PopoverTrigger = React.forwardRef(
  __name(function PopoverTrigger2(props, forwardedRef) {
    const { __scopePopover, ...triggerProps } = props;
    const context = usePopoverContext(TRIGGER_NAME, __scopePopover);
    const popperScope = usePopperScope(__scopePopover);
    const composedTriggerRef = useComposedRefs(forwardedRef, context.triggerRef);
    const trigger = (0, import_jsx_runtime.jsx)(
      Primitive.button,
      {
        type: "button",
        "aria-haspopup": "dialog",
        "aria-expanded": context.open,
        "aria-controls": context.open ? context.contentId : void 0,
        "data-state": getState(context.open),
        ...triggerProps,
        ref: composedTriggerRef,
        onClick: composeEventHandlers(props.onClick, context.onOpenToggle)
      }
    );
    return context.hasCustomAnchor ? trigger : (0, import_jsx_runtime.jsx)(PopperAnchor, { asChild: true, ...popperScope, children: trigger });
  }, "PopoverTrigger")
);
var PORTAL_NAME = "PopoverPortal";
var [PortalProvider, usePortalContext] = createPopoverContext(PORTAL_NAME, {
  forceMount: void 0
});
var PopoverPortal = __name((props) => {
  const { __scopePopover, forceMount, children, container } = props;
  const context = usePopoverContext(PORTAL_NAME, __scopePopover);
  return (0, import_jsx_runtime.jsx)(PortalProvider, { scope: __scopePopover, forceMount, children: (0, import_jsx_runtime.jsx)(Presence, { present: forceMount || context.open, children: (0, import_jsx_runtime.jsx)(Portal, { asChild: true, container, children }) }) });
}, "PopoverPortal");
var CONTENT_NAME = "PopoverContent";
var PopoverContent = React.forwardRef(
  // blank line to reduce diff noise
  __name(function PopoverContent2(props, forwardedRef) {
    const portalContext = usePortalContext(CONTENT_NAME, props.__scopePopover);
    const { forceMount = portalContext.forceMount, ...contentProps } = props;
    const context = usePopoverContext(CONTENT_NAME, props.__scopePopover);
    return (0, import_jsx_runtime.jsx)(Presence, { present: forceMount || context.open, children: context.modal ? (0, import_jsx_runtime.jsx)(PopoverContentModal, { ...contentProps, ref: forwardedRef }) : (0, import_jsx_runtime.jsx)(PopoverContentNonModal, { ...contentProps, ref: forwardedRef }) });
  }, "PopoverContent")
);
var Slot = createSlot("PopoverContent.RemoveScroll");
var PopoverContentModal = React.forwardRef(
  // blank line to reduce diff noise
  __name(function PopoverContentModal2(props, forwardedRef) {
    const context = usePopoverContext(CONTENT_NAME, props.__scopePopover);
    const contentRef = React.useRef(null);
    const composedRefs = useComposedRefs(forwardedRef, contentRef);
    const isRightClickOutsideRef = React.useRef(false);
    const { nodes: branchNodes, registry: branchRegistry } = useFocusScopeBranchRegistry();
    React.useEffect(() => {
      const content = contentRef.current;
      if (content) return hideOthers(content);
    }, []);
    return (0, import_jsx_runtime.jsx)(
      Combination_default,
      {
        as: Slot,
        allowPinchZoom: true,
        shards: React.useMemo(
          () => [contentRef, ...branchNodes.map((node) => ({ current: node }))],
          [contentRef, branchNodes]
        ),
        children: (0, import_jsx_runtime.jsx)(
          PopoverContentImpl,
          {
            ...props,
            ref: composedRefs,
            branchNodes,
            branchRegistry,
            trapFocus: context.open,
            disableOutsidePointerEvents: true,
            onCloseAutoFocus: composeEventHandlers(props.onCloseAutoFocus, (event) => {
              event.preventDefault();
              if (!isRightClickOutsideRef.current) context.triggerRef.current?.focus();
            }),
            onPointerDownOutside: composeEventHandlers(
              props.onPointerDownOutside,
              (event) => {
                const originalEvent = event.detail.originalEvent;
                const ctrlLeftClick = originalEvent.button === 0 && originalEvent.ctrlKey === true;
                const isRightClick = originalEvent.button === 2 || ctrlLeftClick;
                isRightClickOutsideRef.current = isRightClick;
              },
              { checkForDefaultPrevented: false }
            ),
            onFocusOutside: composeEventHandlers(
              props.onFocusOutside,
              (event) => event.preventDefault(),
              { checkForDefaultPrevented: false }
            )
          }
        )
      }
    );
  }, "PopoverContentModal")
);
var PopoverContentNonModal = React.forwardRef(
  // blank line to reduce diff noise
  __name(function PopoverContentNonModal2(props, forwardedRef) {
    const context = usePopoverContext(CONTENT_NAME, props.__scopePopover);
    const hasInteractedOutsideRef = React.useRef(false);
    const hasPointerDownOutsideRef = React.useRef(false);
    return (0, import_jsx_runtime.jsx)(
      PopoverContentImpl,
      {
        ...props,
        ref: forwardedRef,
        trapFocus: false,
        disableOutsidePointerEvents: false,
        onCloseAutoFocus: (event) => {
          props.onCloseAutoFocus?.(event);
          if (!event.defaultPrevented) {
            if (!hasInteractedOutsideRef.current) context.triggerRef.current?.focus();
            event.preventDefault();
          }
          hasInteractedOutsideRef.current = false;
          hasPointerDownOutsideRef.current = false;
        },
        onInteractOutside: (event) => {
          props.onInteractOutside?.(event);
          if (!event.defaultPrevented) {
            hasInteractedOutsideRef.current = true;
            if (event.detail.originalEvent.type === "pointerdown") {
              hasPointerDownOutsideRef.current = true;
            }
          }
          const target = event.target;
          const targetIsTrigger = context.triggerRef.current?.contains(target);
          if (targetIsTrigger) event.preventDefault();
          if (event.detail.originalEvent.type === "focusin" && hasPointerDownOutsideRef.current) {
            event.preventDefault();
          }
        }
      }
    );
  }, "PopoverContentNonModal")
);
var PopoverContentImpl = React.forwardRef(
  // blank line to reduce diff noise
  __name(function PopoverContentImpl2(props, forwardedRef) {
    const {
      __scopePopover,
      trapFocus,
      onOpenAutoFocus,
      onCloseAutoFocus,
      disableOutsidePointerEvents,
      onEscapeKeyDown,
      onPointerDownOutside,
      onFocusOutside,
      onInteractOutside,
      "aria-describedby": ariaDescribedby,
      branchNodes,
      branchRegistry,
      ...contentProps
    } = props;
    const context = usePopoverContext(CONTENT_NAME, __scopePopover);
    const popperScope = usePopperScope(__scopePopover);
    const [contentNode, setContentNode] = React.useState(null);
    const composedRefs = useComposedRefs(forwardedRef, setContentNode);
    useFocusScopeBranch(contentNode);
    useFocusGuards();
    return (0, import_jsx_runtime.jsx)(FocusScopeBranchProvider, { registry: branchRegistry, children: (0, import_jsx_runtime.jsx)(
      FocusScope,
      {
        asChild: true,
        loop: true,
        trapped: trapFocus,
        branches: branchNodes,
        onMountAutoFocus: onOpenAutoFocus,
        onUnmountAutoFocus: onCloseAutoFocus,
        children: (0, import_jsx_runtime.jsx)(
          DismissableLayer,
          {
            asChild: true,
            disableOutsidePointerEvents,
            onInteractOutside,
            onEscapeKeyDown,
            onPointerDownOutside,
            onFocusOutside,
            onDismiss: () => context.onOpenChange(false),
            deferPointerDownOutside: true,
            children: (0, import_jsx_runtime.jsx)(
              PopperContent,
              {
                "data-state": getState(context.open),
                role: "dialog",
                id: context.contentId,
                "aria-labelledby": context.titlePresent ? context.titleId : void 0,
                "aria-describedby": context.descriptionPresent ? concatAriaDescribedby(ariaDescribedby, context.descriptionId) : ariaDescribedby,
                ...popperScope,
                ...contentProps,
                ref: composedRefs,
                style: {
                  ...contentProps.style,
                  // re-namespace exposed content custom properties
                  ...{
                    "--radix-popover-content-transform-origin": "var(--radix-popper-transform-origin)",
                    "--radix-popover-content-available-width": "var(--radix-popper-available-width)",
                    "--radix-popover-content-available-height": "var(--radix-popper-available-height)",
                    "--radix-popover-trigger-width": "var(--radix-popper-anchor-width)",
                    "--radix-popover-trigger-height": "var(--radix-popper-anchor-height)"
                  }
                }
              }
            )
          }
        )
      }
    ) });
  }, "PopoverContentImpl")
);
var PopoverTitle = React.forwardRef(
  __name(function PopoverTitle2(props, forwardedRef) {
    const { __scopePopover, ...titleProps } = props;
    const context = usePopoverContext("PopoverTitle", __scopePopover);
    const { setTitleCount } = context;
    useLayoutEffect2(() => {
      setTitleCount((count) => count + 1);
      return () => setTitleCount((count) => count - 1);
    }, [setTitleCount]);
    return (0, import_jsx_runtime.jsx)(Primitive.h2, { id: context.titleId, ...titleProps, ref: forwardedRef });
  }, "PopoverTitle")
);
var PopoverDescription = React.forwardRef(__name(function PopoverDescription2(props, forwardedRef) {
  const { __scopePopover, ...descriptionProps } = props;
  const context = usePopoverContext("PopoverDescription", __scopePopover);
  const { setDescriptionCount } = context;
  useLayoutEffect2(() => {
    setDescriptionCount((count) => count + 1);
    return () => setDescriptionCount((count) => count - 1);
  }, [setDescriptionCount]);
  return (0, import_jsx_runtime.jsx)(Primitive.p, { id: context.descriptionId, ...descriptionProps, ref: forwardedRef });
}, "PopoverDescription"));
var CLOSE_NAME = "PopoverClose";
var PopoverClose = React.forwardRef(
  __name(function PopoverClose2(props, forwardedRef) {
    const { __scopePopover, ...closeProps } = props;
    const context = usePopoverContext(CLOSE_NAME, __scopePopover);
    return (0, import_jsx_runtime.jsx)(
      Primitive.button,
      {
        type: "button",
        ...closeProps,
        ref: forwardedRef,
        onClick: composeEventHandlers(props.onClick, () => context.onOpenChange(false))
      }
    );
  }, "PopoverClose")
);
var PopoverArrow = React.forwardRef(
  __name(function PopoverArrow2(props, forwardedRef) {
    const { __scopePopover, ...arrowProps } = props;
    const popperScope = usePopperScope(__scopePopover);
    return (0, import_jsx_runtime.jsx)(PopperArrow, { ...popperScope, ...arrowProps, ref: forwardedRef });
  }, "PopoverArrow")
);
function getState(open) {
  return open ? "open" : "closed";
}
__name(getState, "getState");
function concatAriaDescribedby(...values) {
  const ids = /* @__PURE__ */ new Set();
  for (const value of values) {
    if (typeof value !== "string") continue;
    for (const id of String(value).trim().split(/\s+/)) {
      if (id) ids.add(id);
    }
  }
  return ids.size > 0 ? Array.from(ids).join(" ") : void 0;
}
__name(concatAriaDescribedby, "concatAriaDescribedby");
export {
  PopoverAnchor as Anchor,
  PopoverArrow as Arrow,
  PopoverClose as Close,
  PopoverContent as Content,
  PopoverDescription as Description,
  Popover,
  PopoverAnchor,
  PopoverArrow,
  PopoverClose,
  PopoverContent,
  PopoverDescription,
  PopoverPortal,
  PopoverTitle,
  PopoverTrigger,
  PopoverPortal as Portal,
  Popover as Root,
  PopoverTitle as Title,
  PopoverTrigger as Trigger,
  createPopoverScope
};
//# sourceMappingURL=@radix-ui_react-popover.js.map
