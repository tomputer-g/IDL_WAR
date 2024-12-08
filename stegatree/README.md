python eval_stegastamp.py ../../StegaStamp/saved_models/test/ --images_dir ../../StegaStamp/Stega_Tree_Ring_watermarked_Stega\!\!

for file in Stega_Tree_Ring_watermarked_Stega\!\!/*_hidden.png; do
  mv "$file" "${file%_hidden.png}.png"
done

for file in outputs_stegatree/watermarked/*.png; do
  mv "$file" "${file%.png}.jpg"
done