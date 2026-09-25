; TypeScript Tree-sitter queries for chunk boundaries

; Functions and classes
(function_declaration) @definition.function
(class_declaration) @definition.class
(method_definition) @definition.method

; Only capture arrow functions assigned to variables/constants
(variable_declarator
  value: (arrow_function) @definition.fn)

; Or exported arrow functions
(export_statement
  (lexical_declaration
    (variable_declarator
      value: (arrow_function) @definition.fn)))

; Imports only (not all exports, since exports are already captured above)
