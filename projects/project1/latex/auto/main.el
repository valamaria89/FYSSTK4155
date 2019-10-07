(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("revtex4-1" "aps" "reprint")))
   (TeX-run-style-hooks
    "latex2e"
    "Setup/macros"
    "Sections/Frontpage"
    "Sections/Abstract"
    "Sections/Introduction"
    "Sections/Theory"
    "Sections/Method"
    "Sections/Discussion"
    "Sections/Conclusion"
    "revtex4-1"
    "revtex4-110"
    "Setup/style")
   (LaTeX-add-bibliographies
    "bibliography.bib"))
 :latex)

