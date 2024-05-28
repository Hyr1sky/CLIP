# Image Searching with CLIP
---

by Hyr1sky He

## Environment

- clip
- cv2

## Functions

- Matching with similar images in the search image through mathematical methods.
- Matching with images based on your words with CLIP.
- Matching with images based on your target images with CLIP.

## Parameters

```python
filepath = args.dataset_path # your input
searchpath = args.search_path # img to be searched
newfilepath = args.output_path # output
threshold1 = args.threshold1 # fusion similarity threshold
threshold2 = args.threshold2 # final threshold
modelstype = args.model # model name
```

## Run the code

1. Base model\
`python main.py --model base --threshold1 0.7 --thershold2 0.95`

2. text-to-image with CLIP model\
`python main.py --model clip-txt --thershold2 0.8`

3. image-to-image with CLIP model\
`python main.py --model clip-img --thershold2 0.22`
