#!/bin/bash

find . -iname '*.h' \
-o -iname '*.hpp' \
-o -iname '*.cpp' \
-o -iname '*.h.in' \
-o -iname '*.hpp.in' \
-o -iname '*.cpp.in' \
-o -iname '*.cl' \
| grep -v 'build' \
| xargs -n 1 -P 8 -I{} clang-format-3.8 -style=file -i {}
