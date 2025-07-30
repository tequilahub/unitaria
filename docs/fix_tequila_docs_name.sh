curl https://tequilahub.github.io/tequila-tutorials/docs/sphinx/objects.inv -o objects.inv
head -n 4 objects.inv > tequila.inv
tail -n +5 objects.inv | zlib-flate -uncompress | sed -r 's/^tequila_code\.([^ ]*)([^\$]*)\$/tequila.\1\2tequila_code.\1/g' | zlib-flate -compress >> tequila.inv
rm objects.inv
