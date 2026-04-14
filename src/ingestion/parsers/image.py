from pathlib import Path

from .base import BaseParser, ParsedDocument

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


class ImageMetaParser(BaseParser):
    """Parser that generates a 'stub' textual representation for image files.

    Instead of performing OCR (which isn't implemented), it relies purely on the
    filename and the folder structure. Note that images are excluded from actual
    context delivery to the LLM during retrieval via metadata filtering, but
    having them in the vector store allows their names to be searched via UI references.
    """
    def can_parse(self, file_path: Path) -> bool:
        """True for common raster image formats (png/jpg/jpeg/webp/gif)."""
        return file_path.suffix.lower() in IMAGE_EXTENSIONS

    def parse(self, file_path: Path, base_path: Path | None = None) -> list[ParsedDocument]:
        """Convert the image file into a single descriptive ParsedDocument stub.

        The heading hierarchy uses the folder structure leading up to the file,
        climaxing in the filename (without extension) itself.
        """
        source_file = file_path.name
        source_path = str(file_path.resolve())
        name_without_ext = file_path.stem.replace("_", " ").replace("-", " ")

        if base_path is not None:
            try:
                rel = file_path.relative_to(base_path)
                path_parts = list(rel.parts[:-1])  # all folder segments, no filename
            except ValueError:
                path_parts = [file_path.parent.name]
        else:
            path_parts = [file_path.parent.name]

        heading_hierarchy = path_parts + [name_without_ext]
        path_label = " > ".join(path_parts) if path_parts else file_path.parent.name
        content = f"Bild: {name_without_ext} (Pfad: {path_label})"

        return [ParsedDocument(
            content=content,
            source_file=source_file,
            source_path=source_path,
            document_type="image",
            heading_hierarchy=heading_hierarchy,
            metadata={"image_format": file_path.suffix.lower()},
        )]
