"use client";
import {
  VisuallyHidden
} from "./chunk-4SUJ7VX2.js";
import {
  createCollection
} from "./chunk-2NL7CCDF.js";
import {
  useDirection
} from "./chunk-EMHYOJUE.js";
import {
  DismissableLayer
} from "./chunk-ZAPJPNAP.js";
import {
  useCallbackRef
} from "./chunk-X7Z4QOHD.js";
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
  Primitive,
  dispatchDiscreteCustomEvent
} from "./chunk-ZRCZ3OV4.js";
import {
  createSlottable,
  useComposedRefs
} from "./chunk-IEVOZGNX.js";
import {
  require_jsx_runtime
} from "./chunk-XOQ4UB3A.js";
import {
  require_react_dom
} from "./chunk-4U7QFG22.js";
import {
  require_react
} from "./chunk-7KAG3XWH.js";
import {
  __toESM
} from "./chunk-M2PPBBZ4.js";

// node_modules/@radix-ui/react-navigation-menu/dist/index.mjs
var React2 = __toESM(require_react(), 1);
var ReactDOM = __toESM(require_react_dom(), 1);

// node_modules/@radix-ui/react-use-previous/dist/index.mjs
var React = __toESM(require_react(), 1);
var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
function usePrevious(value) {
  const ref = React.useRef({ value, previous: value });
  return React.useMemo(() => {
    if (ref.current.value !== value) {
      ref.current.previous = ref.current.value;
      ref.current.value = value;
    }
    return ref.current.previous;
  }, [value]);
}
__name(usePrevious, "usePrevious");

// node_modules/@radix-ui/react-navigation-menu/dist/index.mjs
var import_jsx_runtime = __toESM(require_jsx_runtime(), 1);
var __defProp2 = Object.defineProperty;
var __name2 = (target, value) => __defProp2(target, "name", { value, configurable: true });
var ActivationMode = {
  Automatic: "automatic",
  Manual: "manual"
};
var Orientation = {
  Vertical: "vertical",
  Horizontal: "horizontal"
};
var Direction = {
  LTR: "ltr",
  RTL: "rtl"
};
var NAVIGATION_MENU_NAME = "NavigationMenu";
var [Collection, useCollection, createCollectionScope] = createCollection(NAVIGATION_MENU_NAME);
var [FocusGroupCollection, useFocusGroupCollection, createFocusGroupCollectionScope] = createCollection(NAVIGATION_MENU_NAME);
var [createNavigationMenuContext, createNavigationMenuScope] = createContextScope(
  NAVIGATION_MENU_NAME,
  [createCollectionScope, createFocusGroupCollectionScope]
);
var [NavigationMenuProviderImpl, useNavigationMenuContext] = createNavigationMenuContext(NAVIGATION_MENU_NAME);
var [ViewportContentProvider, useViewportContentContext] = createNavigationMenuContext(NAVIGATION_MENU_NAME);
var NavigationMenu = React2.forwardRef(
  __name2(function NavigationMenu2(props, forwardedRef) {
    const {
      __scopeNavigationMenu,
      value: valueProp,
      onValueChange,
      defaultValue,
      delayDuration = 200,
      skipDelayDuration = 300,
      orientation = Orientation.Horizontal,
      dir,
      activationMode = ActivationMode.Automatic,
      ...NavigationMenuProps
    } = props;
    const [navigationMenu, setNavigationMenu] = React2.useState(null);
    const composedRef = useComposedRefs(forwardedRef, setNavigationMenu);
    const direction = useDirection(dir);
    const openTimerRef = React2.useRef(0);
    const closeTimerRef = React2.useRef(0);
    const skipDelayTimerRef = React2.useRef(0);
    const [isOpenDelayed, setIsOpenDelayed] = React2.useState(true);
    const [value, setValue] = useControllableState({
      prop: valueProp,
      onChange: __name2((value2) => {
        const isOpen = value2 !== "";
        const hasSkipDelayDuration = skipDelayDuration > 0;
        if (isOpen) {
          window.clearTimeout(skipDelayTimerRef.current);
          if (hasSkipDelayDuration) setIsOpenDelayed(false);
        } else {
          window.clearTimeout(skipDelayTimerRef.current);
          skipDelayTimerRef.current = window.setTimeout(
            () => setIsOpenDelayed(true),
            skipDelayDuration
          );
        }
        onValueChange?.(value2);
      }, "onChange"),
      defaultProp: defaultValue ?? "",
      caller: NAVIGATION_MENU_NAME
    });
    const startCloseTimer = React2.useCallback(() => {
      window.clearTimeout(closeTimerRef.current);
      closeTimerRef.current = window.setTimeout(() => setValue(""), 150);
    }, [setValue]);
    const handleOpen = React2.useCallback(
      (itemValue) => {
        window.clearTimeout(closeTimerRef.current);
        setValue(itemValue);
      },
      [setValue]
    );
    const handleDelayedOpen = React2.useCallback(
      (itemValue) => {
        const isOpenItem = value === itemValue;
        if (isOpenItem) {
          window.clearTimeout(closeTimerRef.current);
        } else {
          openTimerRef.current = window.setTimeout(() => {
            window.clearTimeout(closeTimerRef.current);
            setValue(itemValue);
          }, delayDuration);
        }
      },
      [value, setValue, delayDuration]
    );
    React2.useEffect(() => {
      return () => {
        window.clearTimeout(openTimerRef.current);
        window.clearTimeout(closeTimerRef.current);
        window.clearTimeout(skipDelayTimerRef.current);
      };
    }, []);
    return (0, import_jsx_runtime.jsx)(
      NavigationMenuProvider,
      {
        scope: __scopeNavigationMenu,
        isRootMenu: true,
        value,
        dir: direction,
        orientation,
        activationMode,
        rootNavigationMenu: navigationMenu,
        onTriggerEnter: (itemValue) => {
          if (activationMode === ActivationMode.Automatic) {
            window.clearTimeout(openTimerRef.current);
            if (isOpenDelayed) {
              handleDelayedOpen(itemValue);
            } else {
              handleOpen(itemValue);
            }
          }
        },
        onTriggerLeave: () => {
          if (activationMode === ActivationMode.Automatic) {
            window.clearTimeout(openTimerRef.current);
            startCloseTimer();
          }
        },
        onContentEnter: () => {
          if (activationMode === ActivationMode.Automatic) {
            window.clearTimeout(closeTimerRef.current);
          }
        },
        onContentLeave: () => {
          if (activationMode === ActivationMode.Automatic) {
            startCloseTimer();
          }
        },
        onItemSelect: (itemValue) => {
          setValue((prevValue) => prevValue === itemValue ? "" : itemValue);
        },
        onItemDismiss: () => setValue(""),
        children: (0, import_jsx_runtime.jsx)(
          Primitive.nav,
          {
            "aria-label": "Main",
            "data-orientation": orientation,
            dir: direction,
            ...NavigationMenuProps,
            ref: composedRef
          }
        )
      }
    );
  }, "NavigationMenu")
);
var SUB_NAME = "NavigationMenuSub";
var NavigationMenuSub = React2.forwardRef(
  // blank line to reduce diff noise
  __name2(function NavigationMenuSub2(props, forwardedRef) {
    const {
      __scopeNavigationMenu,
      value: valueProp,
      onValueChange,
      defaultValue,
      orientation = Orientation.Horizontal,
      activationMode: activationModeProp,
      disableToggle = true,
      ...subProps
    } = props;
    const context = useNavigationMenuContext(SUB_NAME, __scopeNavigationMenu);
    const activationMode = activationModeProp ?? context.activationMode;
    const [value, setValue] = useControllableState({
      prop: valueProp,
      onChange: onValueChange,
      defaultProp: defaultValue ?? "",
      caller: SUB_NAME
    });
    return (0, import_jsx_runtime.jsx)(
      NavigationMenuProvider,
      {
        scope: __scopeNavigationMenu,
        isRootMenu: false,
        value,
        dir: context.dir,
        orientation,
        activationMode,
        rootNavigationMenu: context.rootNavigationMenu,
        onTriggerEnter: (itemValue) => {
          if (activationMode === ActivationMode.Automatic) {
            setValue(itemValue);
          }
        },
        onItemSelect: (itemValue) => {
          setValue((prevValue) => disableToggle || prevValue !== itemValue ? itemValue : "");
        },
        onItemDismiss: () => setValue(""),
        children: (0, import_jsx_runtime.jsx)(Primitive.div, { "data-orientation": orientation, ...subProps, ref: forwardedRef })
      }
    );
  }, "NavigationMenuSub")
);
var NavigationMenuProvider = __name2((props) => {
  const {
    scope,
    isRootMenu,
    rootNavigationMenu,
    dir,
    orientation,
    activationMode,
    children,
    value,
    onItemSelect,
    onItemDismiss,
    onTriggerEnter,
    onTriggerLeave,
    onContentEnter,
    onContentLeave
  } = props;
  const [viewport, setViewport] = React2.useState(null);
  const [viewportContent, setViewportContent] = React2.useState(/* @__PURE__ */ new Map());
  const [indicatorTrack, setIndicatorTrack] = React2.useState(null);
  return (0, import_jsx_runtime.jsx)(
    NavigationMenuProviderImpl,
    {
      scope,
      isRootMenu,
      rootNavigationMenu,
      value,
      previousValue: usePrevious(value),
      baseId: useId(),
      dir,
      orientation,
      activationMode,
      viewport,
      onViewportChange: setViewport,
      indicatorTrack,
      onIndicatorTrackChange: setIndicatorTrack,
      onTriggerEnter: useCallbackRef(onTriggerEnter),
      onTriggerLeave: useCallbackRef(onTriggerLeave),
      onContentEnter: useCallbackRef(onContentEnter),
      onContentLeave: useCallbackRef(onContentLeave),
      onItemSelect: useCallbackRef(onItemSelect),
      onItemDismiss: useCallbackRef(onItemDismiss),
      onViewportContentChange: React2.useCallback((contentValue, contentData) => {
        setViewportContent((prevContent) => {
          prevContent.set(contentValue, contentData);
          return new Map(prevContent);
        });
      }, []),
      onViewportContentRemove: React2.useCallback((contentValue) => {
        setViewportContent((prevContent) => {
          if (!prevContent.has(contentValue)) return prevContent;
          prevContent.delete(contentValue);
          return new Map(prevContent);
        });
      }, []),
      children: (0, import_jsx_runtime.jsx)(Collection.Provider, { scope, children: (0, import_jsx_runtime.jsx)(ViewportContentProvider, { scope, items: viewportContent, children }) })
    }
  );
}, "NavigationMenuProvider");
var LIST_NAME = "NavigationMenuList";
var NavigationMenuList = React2.forwardRef(
  // blank line to reduce diff noise
  __name2(function NavigationMenuList2(props, forwardedRef) {
    const { __scopeNavigationMenu, ...listProps } = props;
    const context = useNavigationMenuContext(LIST_NAME, __scopeNavigationMenu);
    const list = (0, import_jsx_runtime.jsx)(Primitive.ul, { "data-orientation": context.orientation, ...listProps, ref: forwardedRef });
    return (0, import_jsx_runtime.jsx)(Primitive.div, { style: { position: "relative" }, ref: context.onIndicatorTrackChange, children: (0, import_jsx_runtime.jsx)(Collection.Slot, { scope: __scopeNavigationMenu, children: context.isRootMenu ? (0, import_jsx_runtime.jsx)(FocusGroup, { asChild: true, children: list }) : list }) });
  }, "NavigationMenuList")
);
var ITEM_NAME = "NavigationMenuItem";
var [NavigationMenuItemContextProvider, useNavigationMenuItemContext] = createNavigationMenuContext(ITEM_NAME);
var NavigationMenuItem = React2.forwardRef(
  // blank line to reduce diff noise
  __name2(function NavigationMenuItem2(props, forwardedRef) {
    const { __scopeNavigationMenu, value: valueProp, ...itemProps } = props;
    const autoValue = useId();
    const value = valueProp || autoValue || "LEGACY_REACT_AUTO_VALUE";
    const contentRef = React2.useRef(null);
    const triggerRef = React2.useRef(null);
    const focusProxyRef = React2.useRef(null);
    const restoreContentTabOrderRef = React2.useRef(() => {
    });
    const wasEscapeCloseRef = React2.useRef(false);
    const handleContentEntry = React2.useCallback((side = "start") => {
      if (contentRef.current) {
        restoreContentTabOrderRef.current();
        const candidates = getTabbableCandidates(contentRef.current);
        if (candidates.length) focusFirst(side === "start" ? candidates : candidates.reverse());
      }
    }, []);
    const handleContentExit = React2.useCallback(() => {
      if (contentRef.current) {
        const candidates = getTabbableCandidates(contentRef.current);
        if (candidates.length) restoreContentTabOrderRef.current = removeFromTabOrder(candidates);
      }
    }, []);
    return (0, import_jsx_runtime.jsx)(
      NavigationMenuItemContextProvider,
      {
        scope: __scopeNavigationMenu,
        value,
        triggerRef,
        contentRef,
        focusProxyRef,
        wasEscapeCloseRef,
        onEntryKeyDown: handleContentEntry,
        onFocusProxyEnter: handleContentEntry,
        onRootContentClose: handleContentExit,
        onContentFocusOutside: handleContentExit,
        children: (0, import_jsx_runtime.jsx)(Primitive.li, { ...itemProps, ref: forwardedRef })
      }
    );
  }, "NavigationMenuItem")
);
var TRIGGER_NAME = "NavigationMenuTrigger";
var NavigationMenuTrigger = React2.forwardRef(__name2(function NavigationMenuTrigger2(props, forwardedRef) {
  const { __scopeNavigationMenu, disabled, ...triggerProps } = props;
  const context = useNavigationMenuContext(TRIGGER_NAME, props.__scopeNavigationMenu);
  const itemContext = useNavigationMenuItemContext(TRIGGER_NAME, props.__scopeNavigationMenu);
  const ref = React2.useRef(null);
  const composedRefs = useComposedRefs(ref, itemContext.triggerRef, forwardedRef);
  const triggerId = makeTriggerId(context.baseId, itemContext.value);
  const contentId = makeContentId(context.baseId, itemContext.value);
  const hasPointerMoveOpenedRef = React2.useRef(false);
  const wasClickCloseRef = React2.useRef(false);
  const open = itemContext.value === context.value;
  return (0, import_jsx_runtime.jsxs)(import_jsx_runtime.Fragment, { children: [
    (0, import_jsx_runtime.jsx)(Collection.ItemSlot, { scope: __scopeNavigationMenu, value: itemContext.value, children: (0, import_jsx_runtime.jsx)(FocusGroupItem, { asChild: true, children: (0, import_jsx_runtime.jsx)(
      Primitive.button,
      {
        id: triggerId,
        disabled,
        "data-disabled": disabled ? "" : void 0,
        "data-state": getOpenState(open),
        "aria-expanded": open,
        "aria-controls": open ? contentId : void 0,
        ...triggerProps,
        ref: composedRefs,
        onPointerEnter: composeEventHandlers(props.onPointerEnter, () => {
          wasClickCloseRef.current = false;
          itemContext.wasEscapeCloseRef.current = false;
        }),
        onPointerMove: composeEventHandlers(
          props.onPointerMove,
          whenMouse(() => {
            if (disabled || wasClickCloseRef.current || itemContext.wasEscapeCloseRef.current || hasPointerMoveOpenedRef.current)
              return;
            context.onTriggerEnter(itemContext.value);
            hasPointerMoveOpenedRef.current = true;
          })
        ),
        onPointerLeave: composeEventHandlers(
          props.onPointerLeave,
          whenMouse(() => {
            if (disabled) return;
            context.onTriggerLeave();
            hasPointerMoveOpenedRef.current = false;
          })
        ),
        onClick: composeEventHandlers(props.onClick, () => {
          context.onItemSelect(itemContext.value);
          wasClickCloseRef.current = open;
        }),
        onKeyDown: composeEventHandlers(props.onKeyDown, (event) => {
          const verticalEntryKey = context.dir === Direction.RTL ? "ArrowLeft" : "ArrowRight";
          const entryKey = { horizontal: "ArrowDown", vertical: verticalEntryKey }[context.orientation];
          if (open && event.key === entryKey) {
            itemContext.onEntryKeyDown();
            event.preventDefault();
          }
        })
      }
    ) }) }),
    open && (0, import_jsx_runtime.jsxs)(import_jsx_runtime.Fragment, { children: [
      (0, import_jsx_runtime.jsx)(
        VisuallyHidden,
        {
          "aria-hidden": true,
          tabIndex: 0,
          ref: itemContext.focusProxyRef,
          onFocus: (event) => {
            const content = itemContext.contentRef.current;
            const prevFocusedElement = event.relatedTarget;
            const wasTriggerFocused = prevFocusedElement === ref.current;
            const wasFocusFromContent = content?.contains(prevFocusedElement);
            if (wasTriggerFocused || !wasFocusFromContent) {
              itemContext.onFocusProxyEnter(wasTriggerFocused ? "start" : "end");
            }
          }
        }
      ),
      context.viewport && (0, import_jsx_runtime.jsx)("span", { "aria-owns": contentId })
    ] })
  ] });
}, "NavigationMenuTrigger"));
var LINK_SELECT = "navigationMenu.linkSelect";
var NavigationMenuLink = React2.forwardRef(
  // blank line to reduce diff noise
  __name2(function NavigationMenuLink2(props, forwardedRef) {
    const { __scopeNavigationMenu, active, onSelect, ...linkProps } = props;
    return (0, import_jsx_runtime.jsx)(FocusGroupItem, { asChild: true, children: (0, import_jsx_runtime.jsx)(
      Primitive.a,
      {
        "data-active": active ? "" : void 0,
        "aria-current": active ? "page" : void 0,
        ...linkProps,
        ref: forwardedRef,
        onClick: composeEventHandlers(
          props.onClick,
          (event) => {
            const target = event.target;
            const linkSelectEvent = new CustomEvent(LINK_SELECT, {
              bubbles: true,
              cancelable: true
            });
            target.addEventListener(LINK_SELECT, (event2) => onSelect?.(event2), { once: true });
            dispatchDiscreteCustomEvent(target, linkSelectEvent);
            if (!linkSelectEvent.defaultPrevented && !event.metaKey) {
              const rootContentDismissEvent = new CustomEvent(ROOT_CONTENT_DISMISS, {
                bubbles: true,
                cancelable: true
              });
              dispatchDiscreteCustomEvent(target, rootContentDismissEvent);
            }
          },
          { checkForDefaultPrevented: false }
        )
      }
    ) });
  }, "NavigationMenuLink")
);
var INDICATOR_NAME = "NavigationMenuIndicator";
var NavigationMenuIndicator = React2.forwardRef(__name2(function NavigationMenuIndicator2(props, forwardedRef) {
  const { forceMount, ...indicatorProps } = props;
  const context = useNavigationMenuContext(INDICATOR_NAME, props.__scopeNavigationMenu);
  const isVisible = Boolean(context.value);
  return context.indicatorTrack ? ReactDOM.createPortal(
    (0, import_jsx_runtime.jsx)(Presence, { present: forceMount || isVisible, children: (0, import_jsx_runtime.jsx)(NavigationMenuIndicatorImpl, { ...indicatorProps, ref: forwardedRef }) }),
    context.indicatorTrack
  ) : null;
}, "NavigationMenuIndicator"));
var NavigationMenuIndicatorImpl = React2.forwardRef(__name2(function NavigationMenuIndicatorImpl2(props, forwardedRef) {
  const { __scopeNavigationMenu, ...indicatorProps } = props;
  const context = useNavigationMenuContext(INDICATOR_NAME, __scopeNavigationMenu);
  const getItems = useCollection(__scopeNavigationMenu);
  const [activeTrigger, setActiveTrigger] = React2.useState(
    null
  );
  const [position, setPosition] = React2.useState(null);
  const isHorizontal = context.orientation === Orientation.Horizontal;
  const isVisible = Boolean(context.value);
  React2.useEffect(() => {
    const items = getItems();
    const triggerNode = items.find((item) => item.value === context.value)?.ref.current;
    if (triggerNode) setActiveTrigger(triggerNode);
  }, [getItems, context.value]);
  const handlePositionChange = __name2(() => {
    if (activeTrigger) {
      setPosition({
        size: isHorizontal ? activeTrigger.offsetWidth : activeTrigger.offsetHeight,
        offset: isHorizontal ? activeTrigger.offsetLeft : activeTrigger.offsetTop
      });
    }
  }, "handlePositionChange");
  useResizeObserver(activeTrigger, handlePositionChange);
  useResizeObserver(context.indicatorTrack, handlePositionChange);
  return position ? (0, import_jsx_runtime.jsx)(
    Primitive.div,
    {
      "aria-hidden": true,
      "data-state": isVisible ? "visible" : "hidden",
      "data-orientation": context.orientation,
      ...indicatorProps,
      ref: forwardedRef,
      style: {
        position: "absolute",
        ...isHorizontal ? {
          left: 0,
          width: position.size + "px",
          transform: `translateX(${position.offset}px)`,
          "--radix-navigation-menu-indicator-translate-x": `${position.offset}px`
        } : {
          top: 0,
          height: position.size + "px",
          transform: `translateY(${position.offset}px)`,
          "--radix-navigation-menu-indicator-translate-y": `${position.offset}px`
        },
        ...indicatorProps.style
      }
    }
  ) : null;
}, "NavigationMenuIndicatorImpl"));
var CONTENT_NAME = "NavigationMenuContent";
var NavigationMenuContent = React2.forwardRef(__name2(function NavigationMenuContent2(props, forwardedRef) {
  const { forceMount, ...contentProps } = props;
  const context = useNavigationMenuContext(CONTENT_NAME, props.__scopeNavigationMenu);
  const itemContext = useNavigationMenuItemContext(CONTENT_NAME, props.__scopeNavigationMenu);
  const composedRefs = useComposedRefs(itemContext.contentRef, forwardedRef);
  const open = itemContext.value === context.value;
  const openState = getOpenState(open);
  const commonProps = {
    value: itemContext.value,
    triggerRef: itemContext.triggerRef,
    focusProxyRef: itemContext.focusProxyRef,
    wasEscapeCloseRef: itemContext.wasEscapeCloseRef,
    onContentFocusOutside: itemContext.onContentFocusOutside,
    onRootContentClose: itemContext.onRootContentClose,
    ...contentProps
  };
  return !context.viewport ? (0, import_jsx_runtime.jsx)(Presence, { present: forceMount || open, children: (0, import_jsx_runtime.jsx)(
    NavigationMenuContentImpl,
    {
      "data-state": openState,
      ...commonProps,
      ref: composedRefs,
      onPointerEnter: composeEventHandlers(props.onPointerEnter, context.onContentEnter),
      onPointerLeave: composeEventHandlers(
        props.onPointerLeave,
        whenMouse(context.onContentLeave)
      ),
      style: {
        // Prevent interaction when animating out
        pointerEvents: !open && context.isRootMenu ? "none" : void 0,
        ...commonProps.style
      }
    }
  ) }) : (0, import_jsx_runtime.jsx)(
    ViewportContentMounter,
    {
      "data-state": openState,
      forceMount,
      ...commonProps,
      ref: composedRefs
    }
  );
}, "NavigationMenuContent"));
var ViewportContentMounter = React2.forwardRef(__name2(function ViewportContentMounter2(props, forwardedRef) {
  const context = useNavigationMenuContext(CONTENT_NAME, props.__scopeNavigationMenu);
  const { onViewportContentChange, onViewportContentRemove } = context;
  useLayoutEffect2(() => {
    onViewportContentChange(props.value, {
      ref: forwardedRef,
      ...props
    });
  }, [props, forwardedRef, onViewportContentChange]);
  useLayoutEffect2(() => {
    return () => onViewportContentRemove(props.value);
  }, [props.value, onViewportContentRemove]);
  return null;
}, "ViewportContentMounter"));
var ROOT_CONTENT_DISMISS = "navigationMenu.rootContentDismiss";
var NavigationMenuContentImpl = React2.forwardRef(__name2(function NavigationMenuContentImpl2(props, forwardedRef) {
  const {
    __scopeNavigationMenu,
    value,
    triggerRef,
    focusProxyRef,
    wasEscapeCloseRef,
    onRootContentClose,
    onContentFocusOutside,
    ...contentProps
  } = props;
  const context = useNavigationMenuContext(CONTENT_NAME, __scopeNavigationMenu);
  const ref = React2.useRef(null);
  const composedRefs = useComposedRefs(ref, forwardedRef);
  const triggerId = makeTriggerId(context.baseId, value);
  const contentId = makeContentId(context.baseId, value);
  const getItems = useCollection(__scopeNavigationMenu);
  const prevMotionAttributeRef = React2.useRef(null);
  const { onItemDismiss } = context;
  React2.useEffect(() => {
    const content = ref.current;
    if (context.isRootMenu && content) {
      const handleClose = __name2(() => {
        onItemDismiss();
        onRootContentClose();
        if (content.contains(document.activeElement)) triggerRef.current?.focus();
      }, "handleClose");
      content.addEventListener(ROOT_CONTENT_DISMISS, handleClose);
      return () => content.removeEventListener(ROOT_CONTENT_DISMISS, handleClose);
    }
  }, [context.isRootMenu, props.value, triggerRef, onItemDismiss, onRootContentClose]);
  const motionAttribute = React2.useMemo(() => {
    const items = getItems();
    const values = items.map((item) => item.value);
    if (context.dir === Direction.RTL) values.reverse();
    const index = values.indexOf(context.value);
    const prevIndex = values.indexOf(context.previousValue);
    const isSelected = value === context.value;
    const wasSelected = prevIndex === values.indexOf(value);
    if (!isSelected && !wasSelected) return prevMotionAttributeRef.current;
    const attribute = (() => {
      if (index !== prevIndex) {
        if (isSelected && prevIndex !== -1) return index > prevIndex ? "from-end" : "from-start";
        if (wasSelected && index !== -1) return index > prevIndex ? "to-start" : "to-end";
      }
      return null;
    })();
    prevMotionAttributeRef.current = attribute;
    return attribute;
  }, [context.previousValue, context.value, context.dir, getItems, value]);
  return (0, import_jsx_runtime.jsx)(FocusGroup, { asChild: true, children: (0, import_jsx_runtime.jsx)(
    DismissableLayer,
    {
      id: contentId,
      "aria-labelledby": triggerId,
      "data-motion": motionAttribute,
      "data-orientation": context.orientation,
      ...contentProps,
      ref: composedRefs,
      disableOutsidePointerEvents: false,
      onDismiss: () => {
        const rootContentDismissEvent = new Event(ROOT_CONTENT_DISMISS, {
          bubbles: true,
          cancelable: true
        });
        ref.current?.dispatchEvent(rootContentDismissEvent);
      },
      onFocusOutside: composeEventHandlers(props.onFocusOutside, (event) => {
        onContentFocusOutside();
        const target = event.target;
        const relatedTarget = event.detail.originalEvent.relatedTarget;
        const focusMovedIntoMenu = context.rootNavigationMenu?.contains(target);
        const focusCameFromMenu = context.rootNavigationMenu?.contains(relatedTarget);
        if (focusMovedIntoMenu || !focusCameFromMenu) {
          event.preventDefault();
        }
      }),
      onPointerDownOutside: composeEventHandlers(props.onPointerDownOutside, (event) => {
        const target = event.target;
        const isTrigger = getItems().some((item) => item.ref.current?.contains(target));
        const isRootViewport = context.isRootMenu && context.viewport?.contains(target);
        if (isTrigger || isRootViewport || !context.isRootMenu) event.preventDefault();
      }),
      onKeyDown: composeEventHandlers(props.onKeyDown, (event) => {
        const isMetaKey = event.altKey || event.ctrlKey || event.metaKey;
        const isTabKey = event.key === "Tab" && !isMetaKey;
        if (isTabKey) {
          const candidates = getTabbableCandidates(event.currentTarget);
          const focusedElement = document.activeElement;
          const index = candidates.findIndex((candidate) => candidate === focusedElement);
          const isMovingBackwards = event.shiftKey;
          const nextCandidates = isMovingBackwards ? candidates.slice(0, index).reverse() : candidates.slice(index + 1, candidates.length);
          if (focusFirst(nextCandidates)) {
            event.preventDefault();
          } else {
            focusProxyRef.current?.focus();
          }
        }
      }),
      onEscapeKeyDown: composeEventHandlers(props.onEscapeKeyDown, (_event) => {
        wasEscapeCloseRef.current = true;
      })
    }
  ) });
}, "NavigationMenuContentImpl"));
var VIEWPORT_NAME = "NavigationMenuViewport";
var Slottable = createSlottable(VIEWPORT_NAME);
var NavigationMenuViewport = React2.forwardRef(__name2(function NavigationMenuViewport2(props, forwardedRef) {
  const { forceMount, ...viewportProps } = props;
  const context = useNavigationMenuContext(VIEWPORT_NAME, props.__scopeNavigationMenu);
  const open = Boolean(context.value);
  return (0, import_jsx_runtime.jsx)(Presence, { present: forceMount || open, children: (0, import_jsx_runtime.jsx)(NavigationMenuViewportImpl, { ...viewportProps, ref: forwardedRef }) });
}, "NavigationMenuViewport"));
var NavigationMenuViewportImpl = React2.forwardRef(__name2(function NavigationMenuViewportImpl2(props, forwardedRef) {
  const { __scopeNavigationMenu, children, ...viewportImplProps } = props;
  const context = useNavigationMenuContext(VIEWPORT_NAME, __scopeNavigationMenu);
  const composedRefs = useComposedRefs(forwardedRef, context.onViewportChange);
  const viewportContentContext = useViewportContentContext(
    CONTENT_NAME,
    props.__scopeNavigationMenu
  );
  const [size, setSize] = React2.useState(null);
  const [content, setContent] = React2.useState(null);
  const viewportWidth = size ? size?.width + "px" : void 0;
  const viewportHeight = size ? size?.height + "px" : void 0;
  const open = Boolean(context.value);
  const activeContentValue = open ? context.value : context.previousValue;
  const handleSizeChange = __name2(() => {
    if (content) setSize({ width: content.offsetWidth, height: content.offsetHeight });
  }, "handleSizeChange");
  useResizeObserver(content, handleSizeChange);
  return (0, import_jsx_runtime.jsxs)(
    Primitive.div,
    {
      "data-state": getOpenState(open),
      "data-orientation": context.orientation,
      ...viewportImplProps,
      ref: composedRefs,
      style: {
        // Prevent interaction when animating out
        pointerEvents: !open && context.isRootMenu ? "none" : void 0,
        "--radix-navigation-menu-viewport-width": viewportWidth,
        "--radix-navigation-menu-viewport-height": viewportHeight,
        ...viewportImplProps.style
      },
      onPointerEnter: composeEventHandlers(props.onPointerEnter, context.onContentEnter),
      onPointerLeave: composeEventHandlers(props.onPointerLeave, whenMouse(context.onContentLeave)),
      children: [
        (0, import_jsx_runtime.jsx)(Slottable, { children }),
        Array.from(viewportContentContext.items).map(([value, { ref, forceMount, ...props2 }]) => {
          const isActive = activeContentValue === value;
          return (0, import_jsx_runtime.jsx)(Presence, { present: forceMount || isActive, children: (0, import_jsx_runtime.jsx)(
            NavigationMenuViewportItem,
            {
              ...props2,
              contentRef: ref,
              isActive,
              onActiveContentChange: setContent
            }
          ) }, value);
        })
      ]
    }
  );
}, "NavigationMenuViewportImpl"));
var NavigationMenuViewportItem = __name2(({
  contentRef,
  isActive,
  onActiveContentChange,
  ...props
}) => {
  const handleContentChange = React2.useCallback(
    (node) => {
      if (isActive && node) {
        onActiveContentChange(node);
      }
    },
    [isActive, onActiveContentChange]
  );
  const composedRefs = useComposedRefs(contentRef, handleContentChange);
  return (0, import_jsx_runtime.jsx)(NavigationMenuContentImpl, { ...props, ref: composedRefs });
}, "NavigationMenuViewportItem");
var FOCUS_GROUP_NAME = "FocusGroup";
var FocusGroup = React2.forwardRef(
  __name2(function FocusGroup2(props, forwardedRef) {
    const { __scopeNavigationMenu, ...groupProps } = props;
    const context = useNavigationMenuContext(FOCUS_GROUP_NAME, __scopeNavigationMenu);
    return (0, import_jsx_runtime.jsx)(FocusGroupCollection.Provider, { scope: __scopeNavigationMenu, children: (0, import_jsx_runtime.jsx)(FocusGroupCollection.Slot, { scope: __scopeNavigationMenu, children: (0, import_jsx_runtime.jsx)(Primitive.div, { dir: context.dir, ...groupProps, ref: forwardedRef }) }) });
  }, "FocusGroup")
);
var ARROW_KEYS = ["ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown"];
var FOCUS_GROUP_ITEM_NAME = "FocusGroupItem";
var FocusGroupItem = React2.forwardRef(
  __name2(function FocusGroupItem2(props, forwardedRef) {
    const { __scopeNavigationMenu, ...groupProps } = props;
    const getItems = useFocusGroupCollection(__scopeNavigationMenu);
    const context = useNavigationMenuContext(FOCUS_GROUP_ITEM_NAME, __scopeNavigationMenu);
    return (0, import_jsx_runtime.jsx)(FocusGroupCollection.ItemSlot, { scope: __scopeNavigationMenu, children: (0, import_jsx_runtime.jsx)(
      Primitive.button,
      {
        ...groupProps,
        ref: forwardedRef,
        onKeyDown: composeEventHandlers(props.onKeyDown, (event) => {
          const isFocusNavigationKey = ["Home", "End", ...ARROW_KEYS].includes(event.key);
          if (isFocusNavigationKey) {
            let candidateNodes = getItems().map((item) => item.ref.current);
            const prevItemKey = context.dir === Direction.RTL ? "ArrowRight" : "ArrowLeft";
            const prevKeys = [prevItemKey, "ArrowUp", "End"];
            if (prevKeys.includes(event.key)) candidateNodes.reverse();
            if (ARROW_KEYS.includes(event.key)) {
              const currentIndex = candidateNodes.indexOf(event.currentTarget);
              candidateNodes = candidateNodes.slice(currentIndex + 1);
            }
            setTimeout(() => focusFirst(candidateNodes));
            event.preventDefault();
          }
        })
      }
    ) });
  }, "FocusGroupItem")
);
function getTabbableCandidates(container) {
  const nodes = [];
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_ELEMENT, {
    acceptNode: __name2((node) => {
      const isHiddenInput = node.tagName === "INPUT" && node.type === "hidden";
      if (node.disabled || node.hidden || isHiddenInput) return NodeFilter.FILTER_SKIP;
      return node.tabIndex >= 0 ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
    }, "acceptNode")
  });
  while (walker.nextNode()) nodes.push(walker.currentNode);
  return nodes;
}
__name2(getTabbableCandidates, "getTabbableCandidates");
function focusFirst(candidates) {
  const previouslyFocusedElement = document.activeElement;
  return candidates.some((candidate) => {
    if (candidate === previouslyFocusedElement) return true;
    candidate.focus();
    return document.activeElement !== previouslyFocusedElement;
  });
}
__name2(focusFirst, "focusFirst");
function removeFromTabOrder(candidates) {
  candidates.forEach((candidate) => {
    candidate.dataset.tabindex = candidate.getAttribute("tabindex") || "";
    candidate.setAttribute("tabindex", "-1");
  });
  return () => {
    candidates.forEach((candidate) => {
      const prevTabIndex = candidate.dataset.tabindex;
      candidate.setAttribute("tabindex", prevTabIndex);
    });
  };
}
__name2(removeFromTabOrder, "removeFromTabOrder");
function useResizeObserver(element, onResize) {
  const handleResize = useCallbackRef(onResize);
  useLayoutEffect2(() => {
    let rAF = 0;
    if (element) {
      const resizeObserver = new ResizeObserver(() => {
        cancelAnimationFrame(rAF);
        rAF = window.requestAnimationFrame(handleResize);
      });
      resizeObserver.observe(element);
      return () => {
        window.cancelAnimationFrame(rAF);
        resizeObserver.unobserve(element);
      };
    }
  }, [element, handleResize]);
}
__name2(useResizeObserver, "useResizeObserver");
function getOpenState(open) {
  return open ? "open" : "closed";
}
__name2(getOpenState, "getOpenState");
function makeTriggerId(baseId, value) {
  return `${baseId}-trigger-${value}`;
}
__name2(makeTriggerId, "makeTriggerId");
function makeContentId(baseId, value) {
  return `${baseId}-content-${value}`;
}
__name2(makeContentId, "makeContentId");
function whenMouse(handler) {
  return (event) => event.pointerType === "mouse" ? handler(event) : void 0;
}
__name2(whenMouse, "whenMouse");
export {
  NavigationMenuContent as Content,
  NavigationMenuIndicator as Indicator,
  NavigationMenuItem as Item,
  NavigationMenuLink as Link,
  NavigationMenuList as List,
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuIndicator,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuSub,
  NavigationMenuTrigger,
  NavigationMenuViewport,
  NavigationMenu as Root,
  NavigationMenuSub as Sub,
  NavigationMenuTrigger as Trigger,
  NavigationMenuViewport as Viewport,
  createNavigationMenuScope
};
//# sourceMappingURL=@radix-ui_react-navigation-menu.js.map
