cd ~/SMNLS/output/

# copies the post-training bert models to seed folders as used for elmo
for size in base large; do
    for i in 0 1 2 3 4; do
        for method in pos pos-snli pos-vua pos-vua-snli snli vua vua-snli; do
            # get the checkpoint for the last stage, e.g. pos-vua-snli -> snli
            IFS='-' read -ra stages <<< "$method"
            stage="${stages[-1]}"

            file="bert-$size-$i/$method/checkpoints/${stage}_epoch20"
            mkdir "$file"
            cp "$file.pt" "$file/0.pt"
        done
    done
done
