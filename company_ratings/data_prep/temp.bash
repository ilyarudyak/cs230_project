for file in 5w/*/*.txt;
    do echo "$file";
    cat "$file" | wc -l;
done