; Rust chunk definitions using tree-sitter queries

; Functions and methods
(function_item) @definition.function

; Types
(struct_item) @definition.struct
(enum_item) @definition.enum
(trait_item) @definition.trait

; Modules and implementations
(impl_item) @module.impl
(mod_item) @module.mod

; Module-level constants and statics
(const_item) @definition.text
(static_item) @definition.text

; Type aliases
(type_item) @definition.text
