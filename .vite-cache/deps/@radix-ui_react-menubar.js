"use client";
import {
  Menu,
  MenuAnchor,
  MenuArrow,
  MenuCheckboxItem,
  MenuContent,
  MenuGroup,
  MenuItem,
  MenuItemIndicator,
  MenuLabel,
  MenuPortal,
  MenuRadioGroup,
  MenuRadioItem,
  MenuSeparator,
  MenuSub,
  MenuSubContent,
  MenuSubTrigger,
  createMenuScope
} from "./chunk-Y3FSNPSL.js";
import {
  RovingFocusGroup,
  RovingFocusGroupItem,
  createRovingFocusGroupScope
} from "./chunk-CD6LTO34.js";
import "./chunk-5GE3UBVK.js";
import "./chunk-O2ZAVM3M.js";
import {
  createCollection
} from "./chunk-2NL7CCDF.js";
import {
  useDirection
} from "./chunk-EMHYOJUE.js";
import "./chunk-QFUDUECV.js";
import "./chunk-KHPBFKCS.js";
import "./chunk-ZAPJPNAP.js";
import "./chunk-X7Z4QOHD.js";
import {
  useId
} from "./chunk-ROQYZUNY.js";
import {
  useControllableState
} from "./chunk-NLDEGCMH.js";
import "./chunk-ETDQP2KF.js";
import {
  composeEventHandlers
} from "./chunk-E24ISM2B.js";
import "./chunk-V4XET5H7.js";
import {
  createContextScope
} from "./chunk-NYJQAXGL.js";
import {
  Primitive
} from "./chunk-ZRCZ3OV4.js";
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

// node_modules/@radix-ui/react-menubar/dist/index.mjs
var React = __toESM(require_react(), 1);
var import_jsx_runtime = __toESM(require_jsx_runtime(), 1);
var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });
var Direction = {
  LTR: "ltr",
  RTL: "rtl"
};
var MENUBAR_NAME = "Menubar";
var [Collection, useCollection, createCollectionScope] = createCollection(MENUBAR_NAME);
var [createMenubarContext, createMenubarScope] = createContextScope(MENUBAR_NAME, [
  createCollectionScope,
  createRovingFocusGroupScope
]);
var useMenuScope = createMenuScope();
var useRovingFocusGroupScope = createRovingFocusGroupScope();
var [MenubarContextProvider, useMenubarContext] = createMenubarContext(MENUBAR_NAME);
var Menubar = React.forwardRef(
  // blank line to reduce diff noise
  __name(function Menubar2(props, forwardedRef) {
    const {
      __scopeMenubar,
      value: valueProp,
      onValueChange,
      defaultValue,
      loop = true,
      dir,
      ...menubarProps
    } = props;
    const direction = useDirection(dir);
    const rovingFocusGroupScope = useRovingFocusGroupScope(__scopeMenubar);
    const [value, setValue] = useControllableState({
      prop: valueProp,
      onChange: onValueChange,
      defaultProp: defaultValue ?? "",
      caller: MENUBAR_NAME
    });
    const [currentTabStopId, setCurrentTabStopId] = React.useState(null);
    return (0, import_jsx_runtime.jsx)(
      MenubarContextProvider,
      {
        scope: __scopeMenubar,
        value,
        onMenuOpen: React.useCallback(
          (value2) => {
            setValue(value2);
            setCurrentTabStopId(value2);
          },
          [setValue]
        ),
        onMenuClose: React.useCallback(() => setValue(""), [setValue]),
        onMenuToggle: React.useCallback(
          (value2) => {
            setValue((prevValue) => prevValue ? "" : value2);
            setCurrentTabStopId(value2);
          },
          [setValue]
        ),
        dir: direction,
        loop,
        children: (0, import_jsx_runtime.jsx)(Collection.Provider, { scope: __scopeMenubar, children: (0, import_jsx_runtime.jsx)(Collection.Slot, { scope: __scopeMenubar, children: (0, import_jsx_runtime.jsx)(
          RovingFocusGroup,
          {
            asChild: true,
            ...rovingFocusGroupScope,
            orientation: "horizontal",
            loop,
            dir: direction,
            currentTabStopId,
            onCurrentTabStopIdChange: setCurrentTabStopId,
            children: (0, import_jsx_runtime.jsx)(Primitive.div, { role: "menubar", ...menubarProps, ref: forwardedRef })
          }
        ) }) })
      }
    );
  }, "Menubar")
);
var MENU_NAME = "MenubarMenu";
var [MenubarMenuProvider, useMenubarMenuContext] = createMenubarContext(MENU_NAME);
var MenubarMenu = __name((props) => {
  const { __scopeMenubar, value: valueProp, ...menuProps } = props;
  const autoValue = useId();
  const value = valueProp || autoValue || "LEGACY_REACT_AUTO_VALUE";
  const context = useMenubarContext(MENU_NAME, __scopeMenubar);
  const menuScope = useMenuScope(__scopeMenubar);
  const triggerRef = React.useRef(null);
  const wasKeyboardTriggerOpenRef = React.useRef(false);
  const open = context.value === value;
  React.useEffect(() => {
    if (!open) wasKeyboardTriggerOpenRef.current = false;
  }, [open]);
  return (0, import_jsx_runtime.jsx)(
    MenubarMenuProvider,
    {
      scope: __scopeMenubar,
      value,
      triggerId: useId(),
      triggerRef,
      contentId: useId(),
      wasKeyboardTriggerOpenRef,
      children: (0, import_jsx_runtime.jsx)(
        Menu,
        {
          ...menuScope,
          open,
          onOpenChange: (open2) => {
            if (!open2) context.onMenuClose();
          },
          modal: false,
          dir: context.dir,
          ...menuProps
        }
      )
    }
  );
}, "MenubarMenu");
var TRIGGER_NAME = "MenubarTrigger";
var MenubarTrigger = React.forwardRef(
  __name(function MenubarTrigger2(props, forwardedRef) {
    const { __scopeMenubar, disabled = false, ...triggerProps } = props;
    const rovingFocusGroupScope = useRovingFocusGroupScope(__scopeMenubar);
    const menuScope = useMenuScope(__scopeMenubar);
    const context = useMenubarContext(TRIGGER_NAME, __scopeMenubar);
    const menuContext = useMenubarMenuContext(TRIGGER_NAME, __scopeMenubar);
    const ref = React.useRef(null);
    const composedRefs = useComposedRefs(forwardedRef, ref, menuContext.triggerRef);
    const [isFocused, setIsFocused] = React.useState(false);
    const open = context.value === menuContext.value;
    return (0, import_jsx_runtime.jsx)(Collection.ItemSlot, { scope: __scopeMenubar, value: menuContext.value, disabled, children: (0, import_jsx_runtime.jsx)(
      RovingFocusGroupItem,
      {
        asChild: true,
        ...rovingFocusGroupScope,
        focusable: !disabled,
        tabStopId: menuContext.value,
        children: (0, import_jsx_runtime.jsx)(MenuAnchor, { asChild: true, ...menuScope, children: (0, import_jsx_runtime.jsx)(
          Primitive.button,
          {
            type: "button",
            role: "menuitem",
            id: menuContext.triggerId,
            "aria-haspopup": "menu",
            "aria-expanded": open,
            "aria-controls": open ? menuContext.contentId : void 0,
            "data-highlighted": isFocused ? "" : void 0,
            "data-state": open ? "open" : "closed",
            "data-disabled": disabled ? "" : void 0,
            disabled,
            ...triggerProps,
            ref: composedRefs,
            onPointerDown: composeEventHandlers(props.onPointerDown, (event) => {
              if (!disabled && event.button === 0 && event.ctrlKey === false) {
                context.onMenuOpen(menuContext.value);
                if (!open) event.preventDefault();
              }
            }),
            onPointerEnter: composeEventHandlers(props.onPointerEnter, () => {
              const menubarOpen = Boolean(context.value);
              if (menubarOpen && !open) {
                context.onMenuOpen(menuContext.value);
                ref.current?.focus();
              }
            }),
            onKeyDown: composeEventHandlers(props.onKeyDown, (event) => {
              if (disabled) return;
              if (["Enter", " "].includes(event.key)) context.onMenuToggle(menuContext.value);
              if (event.key === "ArrowDown") context.onMenuOpen(menuContext.value);
              if (["Enter", " ", "ArrowDown"].includes(event.key)) {
                menuContext.wasKeyboardTriggerOpenRef.current = true;
                event.preventDefault();
              }
            }),
            onFocus: composeEventHandlers(props.onFocus, () => setIsFocused(true)),
            onBlur: composeEventHandlers(props.onBlur, () => setIsFocused(false))
          }
        ) })
      }
    ) });
  }, "MenubarTrigger")
);
var MenubarPortal = __name((props) => {
  const { __scopeMenubar, ...portalProps } = props;
  const menuScope = useMenuScope(__scopeMenubar);
  return (0, import_jsx_runtime.jsx)(MenuPortal, { ...menuScope, ...portalProps });
}, "MenubarPortal");
var CONTENT_NAME = "MenubarContent";
var MenubarContent = React.forwardRef(
  __name(function MenubarContent2(props, forwardedRef) {
    const { __scopeMenubar, align = "start", ...contentProps } = props;
    const menuScope = useMenuScope(__scopeMenubar);
    const context = useMenubarContext(CONTENT_NAME, __scopeMenubar);
    const menuContext = useMenubarMenuContext(CONTENT_NAME, __scopeMenubar);
    const getItems = useCollection(__scopeMenubar);
    const hasInteractedOutsideRef = React.useRef(false);
    return (0, import_jsx_runtime.jsx)(
      MenuContent,
      {
        id: menuContext.contentId,
        "aria-labelledby": menuContext.triggerId,
        "data-radix-menubar-content": "",
        ...menuScope,
        ...contentProps,
        ref: forwardedRef,
        align,
        onCloseAutoFocus: composeEventHandlers(props.onCloseAutoFocus, (event) => {
          const menubarOpen = Boolean(context.value);
          if (!menubarOpen && !hasInteractedOutsideRef.current) {
            menuContext.triggerRef.current?.focus();
          }
          hasInteractedOutsideRef.current = false;
          event.preventDefault();
        }),
        onFocusOutside: composeEventHandlers(props.onFocusOutside, (event) => {
          const target = event.target;
          const isMenubarTrigger = getItems().some((item) => item.ref.current?.contains(target));
          if (isMenubarTrigger) event.preventDefault();
        }),
        onInteractOutside: composeEventHandlers(props.onInteractOutside, () => {
          hasInteractedOutsideRef.current = true;
        }),
        onEntryFocus: (event) => {
          if (!menuContext.wasKeyboardTriggerOpenRef.current) event.preventDefault();
        },
        onKeyDown: composeEventHandlers(
          props.onKeyDown,
          (event) => {
            if (["ArrowRight", "ArrowLeft"].includes(event.key)) {
              const target = event.target;
              const targetIsSubTrigger = target.hasAttribute("data-radix-menubar-subtrigger");
              const isKeyDownInsideSubMenu = target.closest("[data-radix-menubar-content]") !== event.currentTarget;
              const prevMenuKey = context.dir === Direction.RTL ? "ArrowRight" : "ArrowLeft";
              const isPrevKey = prevMenuKey === event.key;
              const isNextKey = !isPrevKey;
              if (isNextKey && targetIsSubTrigger) return;
              if (isKeyDownInsideSubMenu && isPrevKey) return;
              const items = getItems().filter((item) => !item.disabled);
              let candidateValues = items.map((item) => item.value);
              if (isPrevKey) candidateValues.reverse();
              const currentIndex = candidateValues.indexOf(menuContext.value);
              candidateValues = context.loop ? wrapArray(candidateValues, currentIndex + 1) : candidateValues.slice(currentIndex + 1);
              const [nextValue] = candidateValues;
              if (nextValue) context.onMenuOpen(nextValue);
            }
          },
          { checkForDefaultPrevented: false }
        ),
        style: {
          ...props.style,
          // re-namespace exposed content custom properties
          ...{
            "--radix-menubar-content-transform-origin": "var(--radix-popper-transform-origin)",
            "--radix-menubar-content-available-width": "var(--radix-popper-available-width)",
            "--radix-menubar-content-available-height": "var(--radix-popper-available-height)",
            "--radix-menubar-trigger-width": "var(--radix-popper-anchor-width)",
            "--radix-menubar-trigger-height": "var(--radix-popper-anchor-height)"
          }
        }
      }
    );
  }, "MenubarContent")
);
var MenubarGroup = React.forwardRef(
  __name(function MenubarGroup2(props, forwardedRef) {
    const { __scopeMenubar, ...groupProps } = props;
    const menuScope = useMenuScope(__scopeMenubar);
    return (0, import_jsx_runtime.jsx)(MenuGroup, { ...menuScope, ...groupProps, ref: forwardedRef });
  }, "MenubarGroup")
);
var MenubarLabel = React.forwardRef(
  __name(function MenubarLabel2(props, forwardedRef) {
    const { __scopeMenubar, ...labelProps } = props;
    const menuScope = useMenuScope(__scopeMenubar);
    return (0, import_jsx_runtime.jsx)(MenuLabel, { ...menuScope, ...labelProps, ref: forwardedRef });
  }, "MenubarLabel")
);
var MenubarItem = React.forwardRef(
  __name(function MenubarItem2(props, forwardedRef) {
    const { __scopeMenubar, ...itemProps } = props;
    const menuScope = useMenuScope(__scopeMenubar);
    return (0, import_jsx_runtime.jsx)(MenuItem, { ...menuScope, ...itemProps, ref: forwardedRef });
  }, "MenubarItem")
);
var MenubarCheckboxItem = React.forwardRef(
  // blank line to reduce diff noise
  __name(function MenubarCheckboxItem2(props, forwardedRef) {
    const { __scopeMenubar, ...checkboxItemProps } = props;
    const menuScope = useMenuScope(__scopeMenubar);
    return (0, import_jsx_runtime.jsx)(MenuCheckboxItem, { ...menuScope, ...checkboxItemProps, ref: forwardedRef });
  }, "MenubarCheckboxItem")
);
var MenubarRadioGroup = React.forwardRef(
  // blank line to reduce diff noise
  __name(function MenubarRadioGroup2(props, forwardedRef) {
    const { __scopeMenubar, ...radioGroupProps } = props;
    const menuScope = useMenuScope(__scopeMenubar);
    return (0, import_jsx_runtime.jsx)(MenuRadioGroup, { ...menuScope, ...radioGroupProps, ref: forwardedRef });
  }, "MenubarRadioGroup")
);
var MenubarRadioItem = React.forwardRef(
  // blank line to reduce diff noise
  __name(function MenubarRadioItem2(props, forwardedRef) {
    const { __scopeMenubar, ...radioItemProps } = props;
    const menuScope = useMenuScope(__scopeMenubar);
    return (0, import_jsx_runtime.jsx)(MenuRadioItem, { ...menuScope, ...radioItemProps, ref: forwardedRef });
  }, "MenubarRadioItem")
);
var MenubarItemIndicator = React.forwardRef(__name(function MenubarItemIndicator2(props, forwardedRef) {
  const { __scopeMenubar, ...itemIndicatorProps } = props;
  const menuScope = useMenuScope(__scopeMenubar);
  return (0, import_jsx_runtime.jsx)(MenuItemIndicator, { ...menuScope, ...itemIndicatorProps, ref: forwardedRef });
}, "MenubarItemIndicator"));
var MenubarSeparator = React.forwardRef(
  // blank line to reduce diff noise
  __name(function MenubarSeparator2(props, forwardedRef) {
    const { __scopeMenubar, ...separatorProps } = props;
    const menuScope = useMenuScope(__scopeMenubar);
    return (0, import_jsx_runtime.jsx)(MenuSeparator, { ...menuScope, ...separatorProps, ref: forwardedRef });
  }, "MenubarSeparator")
);
var MenubarArrow = React.forwardRef(
  __name(function MenubarArrow2(props, forwardedRef) {
    const { __scopeMenubar, ...arrowProps } = props;
    const menuScope = useMenuScope(__scopeMenubar);
    return (0, import_jsx_runtime.jsx)(MenuArrow, { ...menuScope, ...arrowProps, ref: forwardedRef });
  }, "MenubarArrow")
);
var SUB_NAME = "MenubarSub";
var MenubarSub = __name((props) => {
  const { __scopeMenubar, children, open: openProp, onOpenChange, defaultOpen } = props;
  const menuScope = useMenuScope(__scopeMenubar);
  const [open, setOpen] = useControllableState({
    prop: openProp,
    defaultProp: defaultOpen ?? false,
    onChange: onOpenChange,
    caller: SUB_NAME
  });
  return (0, import_jsx_runtime.jsx)(MenuSub, { ...menuScope, open, onOpenChange: setOpen, children });
}, "MenubarSub");
var MenubarSubTrigger = React.forwardRef(
  // blank line to reduce diff noise
  __name(function MenubarSubTrigger2(props, forwardedRef) {
    const { __scopeMenubar, ...subTriggerProps } = props;
    const menuScope = useMenuScope(__scopeMenubar);
    return (0, import_jsx_runtime.jsx)(
      MenuSubTrigger,
      {
        "data-radix-menubar-subtrigger": "",
        ...menuScope,
        ...subTriggerProps,
        ref: forwardedRef
      }
    );
  }, "MenubarSubTrigger")
);
var MenubarSubContent = React.forwardRef(
  // blank line to reduce diff noise
  __name(function MenubarSubContent2(props, forwardedRef) {
    const { __scopeMenubar, ...subContentProps } = props;
    const menuScope = useMenuScope(__scopeMenubar);
    return (0, import_jsx_runtime.jsx)(
      MenuSubContent,
      {
        ...menuScope,
        "data-radix-menubar-content": "",
        ...subContentProps,
        ref: forwardedRef,
        style: {
          ...props.style,
          // re-namespace exposed content custom properties
          ...{
            "--radix-menubar-content-transform-origin": "var(--radix-popper-transform-origin)",
            "--radix-menubar-content-available-width": "var(--radix-popper-available-width)",
            "--radix-menubar-content-available-height": "var(--radix-popper-available-height)",
            "--radix-menubar-trigger-width": "var(--radix-popper-anchor-width)",
            "--radix-menubar-trigger-height": "var(--radix-popper-anchor-height)"
          }
        }
      }
    );
  }, "MenubarSubContent")
);
function wrapArray(array, startIndex) {
  return array.map((_, index) => array[(startIndex + index) % array.length]);
}
__name(wrapArray, "wrapArray");
export {
  MenubarArrow as Arrow,
  MenubarCheckboxItem as CheckboxItem,
  MenubarContent as Content,
  MenubarGroup as Group,
  MenubarItem as Item,
  MenubarItemIndicator as ItemIndicator,
  MenubarLabel as Label,
  MenubarMenu as Menu,
  Menubar,
  MenubarArrow,
  MenubarCheckboxItem,
  MenubarContent,
  MenubarGroup,
  MenubarItem,
  MenubarItemIndicator,
  MenubarLabel,
  MenubarMenu,
  MenubarPortal,
  MenubarRadioGroup,
  MenubarRadioItem,
  MenubarSeparator,
  MenubarSub,
  MenubarSubContent,
  MenubarSubTrigger,
  MenubarTrigger,
  MenubarPortal as Portal,
  MenubarRadioGroup as RadioGroup,
  MenubarRadioItem as RadioItem,
  Menubar as Root,
  MenubarSeparator as Separator,
  MenubarSub as Sub,
  MenubarSubContent as SubContent,
  MenubarSubTrigger as SubTrigger,
  MenubarTrigger as Trigger,
  createMenubarScope
};
//# sourceMappingURL=@radix-ui_react-menubar.js.map
