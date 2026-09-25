; Go chunk definitions

; Functions and methods
(function_declaration) @definition.function
(method_declaration) @definition.method

; Types
(type_declaration) @definition.class

; Top-level declarations
(source_file (var_declaration) @module)
(source_file (const_declaration) @module)

; Imports and package
