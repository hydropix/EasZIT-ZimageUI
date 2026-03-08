"""
Pre-built workflow templates
"""

from .builder import create_txt2img_workflow, create_img2img_workflow, create_inpainting_workflow


def get_txt2img_workflow(**kwargs):
    """Get standard txt2img workflow."""
    return create_txt2img_workflow(**kwargs)


def get_img2img_workflow(**kwargs):
    """Get standard img2img workflow."""
    return create_img2img_workflow(**kwargs)


def get_inpainting_workflow(**kwargs):
    """Get standard inpainting workflow."""
    return create_inpainting_workflow(**kwargs)
