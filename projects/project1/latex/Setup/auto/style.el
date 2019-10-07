(TeX-add-style-hook
 "style"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("babel" "norsk" "english") ("parskip" "parfill") ("geometry" "margin=0.9in") ("microtype" "final") ("nth" "super") ("tikz-feynman" "compat=1.1.0") ("caption" "font={scriptsize}") ("mdframed" "framemethod=TikZ") ("tcolorbox" "listings" "theorems" "skins" "breakable") ("cleveref" "nameinlink")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-environments-local "minted")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "standalone"
    "inputenc"
    "babel"
    "parskip"
    "geometry"
    "lmodern"
    "microtype"
    "enumitem"
    "siunitx"
    "nth"
    "fancyhdr"
    "simpler-wick"
    "physics"
    "xparse"
    "braket"
    "tikz-feynman"
    "graphicx"
    "subfig"
    "float"
    "multirow"
    "array"
    "caption"
    "chngcntr"
    "booktabs"
    "tikz"
    "mathtools"
    "amssymb"
    "xfrac"
    "bm"
    "bbm"
    "mdframed"
    "tcolorbox"
    "hyperref"
    "cleveref"
    "algorithm"
    "algpseudocode"
    "algpascal"
    "minted"
    "listings")
   (TeX-add-symbols
    "frontmatter"
    "mainmatter"
    "algorithmautorefname")
   (LaTeX-add-mdframed-mdfdefinestyles
    "MyFrame")
   (LaTeX-add-xcolor-definecolors
    "codegreen"
    "codegray"
    "codepurple"
    "backcolour")
   (LaTeX-add-listings-lstdefinestyles
    "mystyle"))
 :latex)

