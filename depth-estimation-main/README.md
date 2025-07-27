# Navigation standalone module
Based on UDepth and UWDepth models.

## UDepth (Slower but more accurate):
https://github.com/user-attachments/assets/20039f55-194a-4779-aea9-57e3b4e75557

## UWDepth (Faster but less accurate):
https://github.com/user-attachments/assets/e64a0735-8fad-428a-a8fa-8982a85e3c8e

Default Depth type is `type:=udepth``, but can be changed via `type:=uw_depth`

Provides a node /depth/map for depth estimation, and custom tensor interface for depth libraries in utils

Note: UDepth and UW_Depth are included implicitly as they contain extensive code changes to speed up inference time and increase GPU usage
