"""
Annotation format converter for bird detection project.

Supports conversion between CVAT XML, COCO JSON, and YOLO formats.
All six pairwise conversions are supported:
  cvat_to_yolo, cvat_to_coco
  coco_to_yolo, coco_to_cvat
  yolo_to_coco, yolo_to_cvat
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from xml.dom import minidom


class AnnotationConverter:
    """Convert between annotation formats CVAT XML, COCO JSON, and YOLO."""
    def __init__(self, class_mapping: Optional[Dict[str, int]] = None):
        """
        Initialize the annotation converter.

        Args:
            class_mapping: Optional dictionary mapping class names to YOLO class IDs (0-indexed).
                          If None, class IDs will be auto-generated in order of appearance.
        """
        self.class_mapping = class_mapping or {}
        self._auto_class_id = 0

    def _get_or_create_class_id(self, class_name: str) -> int:
        """Get class ID for a class name, creating if necessary."""
        if class_name not in self.class_mapping:
            self.class_mapping[class_name] = self._auto_class_id
            self._auto_class_id += 1
        return self.class_mapping[class_name]

    def _normalize_bbox(self,
                       x_min: float, y_min: float,
                       x_max: float, y_max: float,
                       img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert absolute bounding box to YOLO format (normalized center coords).

        Args:
            x_min, y_min: Top-left corner (absolute pixels)
            x_max, y_max: Bottom-right corner (absolute pixels)
            img_width, img_height: Image dimensions

        Returns:
            Tuple of (x_center, y_center, width, height) normalized to [0, 1]
        """
        # Calculate center and dimensions
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + width / 2.0
        y_center = y_min + height / 2.0

        # Normalize by image dimensions
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height

        return x_center_norm, y_center_norm, width_norm, height_norm

    def _denormalize_bbox(self,
                         x_center: float, y_center: float,
                         width: float, height: float,
                         img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert YOLO normalized bbox to absolute pixel coordinates.

        Args:
            x_center, y_center, width, height: Normalized YOLO bbox values [0, 1]
            img_width, img_height: Image dimensions in pixels

        Returns:
            Tuple of (xtl, ytl, xbr, ybr) in absolute pixels
        """
        w_abs = width * img_width
        h_abs = height * img_height
        xtl = x_center * img_width - w_abs / 2.0
        ytl = y_center * img_height - h_abs / 2.0
        return xtl, ytl, xtl + w_abs, ytl + h_abs

    def _build_coco_output(self,
                           images: List[dict],
                           annotations: List[dict],
                           categories: List[dict]) -> dict:
        """Assemble a COCO-format dict from image, annotation, and category lists."""
        return {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

    def _build_cvat_xml(self, image_dicts: List[dict]) -> str:
        """
        Build a pretty-printed CVAT XML string.

        Args:
            image_dicts: List of dicts with keys:
                filename (str), width (int), height (int),
                boxes (list of {label, xtl, ytl, xbr, ybr})

        Returns:
            Pretty-printed XML string
        """
        root = ET.Element("annotations")

        for img in image_dicts:
            img_elem = ET.SubElement(root, "image",
                                     name=img["filename"],
                                     width=str(img["width"]),
                                     height=str(img["height"]))
            for box in img["boxes"]:
                ET.SubElement(img_elem, "box",
                              label=box["label"],
                              xtl=f"{box['xtl']:.2f}",
                              ytl=f"{box['ytl']:.2f}",
                              xbr=f"{box['xbr']:.2f}",
                              ybr=f"{box['ybr']:.2f}")

        xml_str = ET.tostring(root, encoding="unicode")
        return minidom.parseString(xml_str).toprettyxml(indent="  ")

    def cvat_to_yolo(self,
                     cvat_xml_path: Union[str, Path],
                     output_dir: Union[str, Path],
                     image_name_pattern: str = "frame_{:06d}") -> Dict[str, int]:
        """
        Convert CVAT XML annotations to YOLO format.

        CVAT XML contains video annotations with tracks (temporal sequences of bounding boxes).
        Each frame with annotations will generate a separate YOLO label file.

        Args:
            cvat_xml_path: Path to CVAT XML file
            output_dir: Directory to save YOLO label files
            image_name_pattern: Pattern for generating image names from frame numbers.
                               Default: "frame_{:06d}" produces "frame_000001", "frame_000002", etc.

        Returns:
            Dictionary mapping class names to their assigned YOLO class IDs

        Note:
            - Bounding boxes are converted from (xtl, ytl, xbr, ybr) to normalized center format
            - Image dimensions are extracted from task metadata
            - Label files are named to match image files (e.g., frame_000001.txt)
        """
        cvat_xml_path = Path(cvat_xml_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse XML
        tree = ET.parse(cvat_xml_path)
        root = tree.getroot()

        # Extract image dimensions from tasks
        task_dimensions = {}
        for task in root.findall('.//task'):
            task_id = task.find('id').text
            original_size = task.find('original_size')
            if original_size is not None:
                width = int(original_size.find('width').text)
                height = int(original_size.find('height').text)
                task_dimensions[task_id] = (width, height)

        # Group annotations by frame
        frame_annotations = {}  # {frame_num: [(class_id, bbox), ...]}

        # Process all tracks
        for track in root.findall('.//track'):
            label = track.get('label')
            task_id = track.get('task_id')
            class_id = self._get_or_create_class_id(label)

            # Get image dimensions for this task
            if task_id not in task_dimensions:
                print(f"Warning: No dimensions found for task_id {task_id}, skipping track")
                continue
            img_width, img_height = task_dimensions[task_id]

            # Process each box in the track
            for box in track.findall('box'):
                frame = int(box.get('frame'))
                outside = box.get('outside', '0')

                # Skip boxes marked as outside
                if outside == '1':
                    continue

                # Extract bounding box coordinates
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))

                # Convert to YOLO format
                x_center, y_center, width, height = self._normalize_bbox(
                    xtl, ytl, xbr, ybr, img_width, img_height
                )

                # Add to frame annotations
                if frame not in frame_annotations:
                    frame_annotations[frame] = []
                frame_annotations[frame].append((class_id, x_center, y_center, width, height))

        # Write label files for each frame
        for frame_num, annotations in frame_annotations.items():
            # Generate label filename
            image_name = image_name_pattern.format(frame_num)
            label_path = output_dir / f"{image_name}.txt"

            # Write annotations
            with open(label_path, 'w') as f:
                for class_id, x_center, y_center, width, height in annotations:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        print(f"Converted {len(frame_annotations)} frames from CVAT to YOLO format")
        print(f"Output directory: {output_dir}")

        return self.class_mapping

    def coco_to_yolo(self,
                     coco_json_path: Union[str, Path],
                     output_dir: Union[str, Path],
                     use_filename: bool = True) -> Dict[str, int]:
        """
        Convert COCO JSON annotations to YOLO format.

        Args:
            coco_json_path: Path to COCO JSON file
            output_dir: Directory to save YOLO label files
            use_filename: If True, use the image filename from COCO (without extension) for label files.
                         If False, use image_id as the label filename.

        Returns:
            Dictionary mapping class names to their assigned YOLO class IDs

        Note:
            - COCO bounding boxes [x, y, width, height] are converted to YOLO normalized center format
            - COCO category IDs are mapped to 0-indexed YOLO class IDs based on class_mapping
            - Label files are named to match image files (e.g., frame_000001.txt for frame_000001.png)
        """
        coco_json_path = Path(coco_json_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load COCO JSON
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        # Build category mapping: COCO category_id -> class_name
        coco_categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

        # Build image info mapping: image_id -> (filename, width, height)
        image_info = {
            img['id']: (img['file_name'], img['width'], img['height'])
            for img in coco_data['images']
        }

        # Group annotations by image_id
        image_annotations = {}  # {image_id: [(class_id, bbox), ...]}

        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            category_id = ann['category_id']

            # Get class name and YOLO class ID
            class_name = coco_categories.get(category_id, 'unknown')
            class_id = self._get_or_create_class_id(class_name)

            # Get image dimensions
            if image_id not in image_info:
                print(f"Warning: Image ID {image_id} not found in images, skipping annotation")
                continue

            filename, img_width, img_height = image_info[image_id]

            # Extract COCO bbox [x, y, width, height]
            bbox = ann['bbox']
            x_min = bbox[0]
            y_min = bbox[1]
            box_width = bbox[2]
            box_height = bbox[3]
            x_max = x_min + box_width
            y_max = y_min + box_height

            # Convert to YOLO format
            x_center, y_center, width, height = self._normalize_bbox(
                x_min, y_min, x_max, y_max, img_width, img_height
            )

            # Add to image annotations
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append((class_id, x_center, y_center, width, height))

        # Write label files for each image
        for image_id, annotations in image_annotations.items():
            filename, _, _ = image_info[image_id]

            # Generate label filename
            if use_filename:
                # Use image filename without extension
                label_name = Path(filename).stem
            else:
                # Use image ID
                label_name = f"image_{image_id:06d}"

            label_path = output_dir / f"{label_name}.txt"

            # Write annotations
            with open(label_path, 'w') as f:
                for class_id, x_center, y_center, width, height in annotations:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        print(f"Converted {len(image_annotations)} images from COCO to YOLO format")
        print(f"Output directory: {output_dir}")

        return self.class_mapping

    def cvat_to_coco(self,
                     cvat_xml_path: Union[str, Path],
                     output_path: Union[str, Path],
                     image_name_pattern: str = "frame_{:06d}.png") -> Dict[str, int]:
        """
        Convert CVAT XML (track-based) annotations to COCO JSON format.

        Args:
            cvat_xml_path: Path to CVAT XML file
            output_path: Path for the output COCO JSON file
            image_name_pattern: Pattern for image filenames including extension.
                               Default: "frame_{:06d}.png"

        Returns:
            Dictionary mapping class names to their assigned YOLO class IDs
        """
        cvat_xml_path = Path(cvat_xml_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tree = ET.parse(cvat_xml_path)
        root = tree.getroot()

        # Extract image dimensions from tasks
        task_dimensions = {}
        for task in root.findall('.//task'):
            task_id = task.find('id').text
            original_size = task.find('original_size')
            if original_size is not None:
                width = int(original_size.find('width').text)
                height = int(original_size.find('height').text)
                task_dimensions[task_id] = (width, height)

        # Accumulate per-frame data
        frame_info: Dict[int, dict] = {}  # frame_num -> {width, height}
        frame_boxes: Dict[int, list] = {}  # frame_num -> [{label, xtl, ytl, xbr, ybr}]

        for track in root.findall('.//track'):
            label = track.get('label')
            task_id = track.get('task_id')
            self._get_or_create_class_id(label)

            if task_id not in task_dimensions:
                print(f"Warning: No dimensions for task_id {task_id}, skipping track")
                continue
            img_width, img_height = task_dimensions[task_id]

            for box in track.findall('box'):
                frame = int(box.get('frame'))
                if box.get('outside', '0') == '1':
                    continue

                frame_info[frame] = {'width': img_width, 'height': img_height}
                frame_boxes.setdefault(frame, []).append({
                    'label': label,
                    'xtl': float(box.get('xtl')),
                    'ytl': float(box.get('ytl')),
                    'xbr': float(box.get('xbr')),
                    'ybr': float(box.get('ybr')),
                })

        # Build COCO structures
        # Categories: sort by class_id for stable output
        id_to_name = {v: k for k, v in self.class_mapping.items()}
        categories = [
            {'id': class_id + 1, 'name': id_to_name[class_id]}
            for class_id in sorted(id_to_name)
        ]
        name_to_coco_cat = {cat['name']: cat['id'] for cat in categories}

        images = []
        annotations = []
        ann_id = 1

        for frame_num in sorted(frame_info):
            info = frame_info[frame_num]
            file_name = image_name_pattern.format(frame_num)
            images.append({
                'id': frame_num,
                'file_name': file_name,
                'width': info['width'],
                'height': info['height'],
            })
            for box in frame_boxes.get(frame_num, []):
                xtl, ytl, xbr, ybr = box['xtl'], box['ytl'], box['xbr'], box['ybr']
                annotations.append({
                    'id': ann_id,
                    'image_id': frame_num,
                    'category_id': name_to_coco_cat[box['label']],
                    'bbox': [xtl, ytl, xbr - xtl, ybr - ytl],
                    'area': (xbr - xtl) * (ybr - ytl),
                    'iscrowd': 0,
                })
                ann_id += 1

        coco_output = self._build_coco_output(images, annotations, categories)
        with open(output_path, 'w') as f:
            json.dump(coco_output, f, indent=2)

        print(f"Converted {len(images)} frames from CVAT to COCO format")
        print(f"Output file: {output_path}")
        return self.class_mapping

    def coco_to_cvat(self,
                     coco_json_path: Union[str, Path],
                     output_path: Union[str, Path]) -> None:
        """
        Convert COCO JSON annotations to CVAT XML (image-based) format.

        Args:
            coco_json_path: Path to COCO JSON file
            output_path: Path for the output CVAT XML file
        """
        coco_json_path = Path(coco_json_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        coco_categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        image_info = {
            img['id']: {'file_name': img['file_name'],
                        'width': img['width'],
                        'height': img['height']}
            for img in coco_data['images']
        }

        # Group boxes by image
        image_boxes: Dict[int, list] = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_info:
                print(f"Warning: image_id {image_id} not in images, skipping")
                continue
            label = coco_categories.get(ann['category_id'], 'unknown')
            x, y, w, h = ann['bbox']
            image_boxes.setdefault(image_id, []).append({
                'label': label,
                'xtl': x,
                'ytl': y,
                'xbr': x + w,
                'ybr': y + h,
            })

        image_dicts = []
        for image_id, info in image_info.items():
            image_dicts.append({
                'filename': info['file_name'],
                'width': info['width'],
                'height': info['height'],
                'boxes': image_boxes.get(image_id, []),
            })

        xml_str = self._build_cvat_xml(image_dicts)
        with open(output_path, 'w') as f:
            f.write(xml_str)

        print(f"Converted {len(image_dicts)} images from COCO to CVAT format")
        print(f"Output file: {output_path}")

    def yolo_to_coco(self,
                     labels_dir: Union[str, Path],
                     output_path: Union[str, Path],
                     image_size: Tuple[int, int],
                     class_names: Dict[int, str],
                     image_ext: str = ".png") -> Dict[str, int]:
        """
        Convert YOLO label files to COCO JSON format.

        Args:
            labels_dir: Directory containing YOLO .txt label files
            output_path: Path for the output COCO JSON file
            image_size: (width, height) in pixels — applied to all images
            class_names: Mapping of YOLO class_id -> class_name (required)
            image_ext: Image file extension (default ".png")

        Returns:
            Dictionary mapping class names to their assigned YOLO class IDs
        """
        labels_dir = Path(labels_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img_width, img_height = image_size

        # Register all classes so class_mapping is populated
        for class_id, name in class_names.items():
            if name not in self.class_mapping:
                self.class_mapping[name] = class_id

        categories = [
            {'id': class_id + 1, 'name': class_names[class_id]}
            for class_id in sorted(class_names)
        ]

        images = []
        annotations = []
        ann_id = 1
        image_id = 1

        for label_path in sorted(labels_dir.glob("*.txt")):
            stem = label_path.stem
            file_name = stem + image_ext
            images.append({
                'id': image_id,
                'file_name': file_name,
                'width': img_width,
                'height': img_height,
            })

            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    xtl, ytl, xbr, ybr = self._denormalize_bbox(
                        x_center, y_center, width, height, img_width, img_height
                    )
                    annotations.append({
                        'id': ann_id,
                        'image_id': image_id,
                        'category_id': class_id + 1,
                        'bbox': [xtl, ytl, xbr - xtl, ybr - ytl],
                        'area': (xbr - xtl) * (ybr - ytl),
                        'iscrowd': 0,
                    })
                    ann_id += 1

            image_id += 1

        coco_output = self._build_coco_output(images, annotations, categories)
        with open(output_path, 'w') as f:
            json.dump(coco_output, f, indent=2)

        print(f"Converted {len(images)} label files from YOLO to COCO format")
        print(f"Output file: {output_path}")
        return self.class_mapping

    def yolo_to_cvat(self,
                     labels_dir: Union[str, Path],
                     output_path: Union[str, Path],
                     image_size: Tuple[int, int],
                     class_names: Dict[int, str],
                     image_ext: str = ".png") -> None:
        """
        Convert YOLO label files to CVAT XML (image-based) format.

        Args:
            labels_dir: Directory containing YOLO .txt label files
            output_path: Path for the output CVAT XML file
            image_size: (width, height) in pixels — applied to all images
            class_names: Mapping of YOLO class_id -> class_name (required)
            image_ext: Image file extension (default ".png")
        """
        labels_dir = Path(labels_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img_width, img_height = image_size

        # Register all classes
        for class_id, name in class_names.items():
            if name not in self.class_mapping:
                self.class_mapping[name] = class_id

        image_dicts = []

        for label_path in sorted(labels_dir.glob("*.txt")):
            stem = label_path.stem
            file_name = stem + image_ext
            boxes = []

            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    xtl, ytl, xbr, ybr = self._denormalize_bbox(
                        x_center, y_center, width, height, img_width, img_height
                    )
                    boxes.append({
                        'label': class_names[class_id],
                        'xtl': xtl, 'ytl': ytl,
                        'xbr': xbr, 'ybr': ybr,
                    })

            image_dicts.append({
                'filename': file_name,
                'width': img_width,
                'height': img_height,
                'boxes': boxes,
            })

        xml_str = self._build_cvat_xml(image_dicts)
        with open(output_path, 'w') as f:
            f.write(xml_str)

        print(f"Converted {len(image_dicts)} label files from YOLO to CVAT format")
        print(f"Output file: {output_path}")

    def create_yaml_config(self,
                          output_dir: Union[str, Path],
                          dataset_path: str,
                          train_dir: str = "images/train",
                          val_dir: str = "images/val",
                          test_dir: Optional[str] = "images/test",
                          class_mapping: Optional[Dict[str, int]] = None) -> None:
        """
        Create a YOLO dataset configuration YAML file.

        Args:
            output_path: Path to save the YAML config file
            dataset_path: Root path of the dataset (relative or absolute)
            train_dir: Path to training images (relative to dataset_path)
            val_dir: Path to validation images (relative to dataset_path)
            test_dir: Path to test images (relative to dataset_path), or None if no test set
            class_mapping: Dictionary mapping class names to class IDs.
                          If None, uses the converter's current class_mapping.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir / 'yolo.yaml')
        class_map = class_mapping or self.class_mapping

        # Create YAML content
        yaml_content = [
            "# YOLO Dataset Configuration",
            "# Auto-generated by AnnotationConverter",
            "",
            f"path: {dataset_path}  # dataset root dir",
            f"train: {train_dir}  # train images (relative to 'path')",
            f"val: {val_dir}  # val images (relative to 'path')",
        ]

        if test_dir:
            yaml_content.append(f"test: {test_dir}  # test images (relative to 'path')")

        yaml_content.extend([
            "",
            "# Classes",
            "names:"
        ])

        # Sort by class ID and add to YAML
        sorted_classes = sorted(class_map.items(), key=lambda x: x[1])
        for class_name, class_id in sorted_classes:
            yaml_content.append(f"  {class_id}: {class_name}")

        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(yaml_content) + '\n')

        print(f"Created YOLO config: {output_path}")
        print(f"Classes: {dict(sorted_classes)}")


def main():
    """
    Example usage of AnnotationConverter.

    Supported conversions:
        cvat_to_yolo   - CVAT XML (track-based)  → YOLO label files
        cvat_to_coco   - CVAT XML (track-based)  → COCO JSON
        coco_to_yolo   - COCO JSON               → YOLO label files
        coco_to_cvat   - COCO JSON               → CVAT XML (image-based)
        yolo_to_coco   - YOLO label files        → COCO JSON
        yolo_to_cvat   - YOLO label files        → CVAT XML (image-based)
    """
    converter = AnnotationConverter(class_mapping={'Bird': 0, 'bird': 0})

    # Example 1: CVAT → YOLO
    # converter.cvat_to_yolo(
    #     cvat_xml_path='annotations/annotations.xml',
    #     output_dir='data/labels/train',
    #     image_name_pattern='frame_{:06d}'
    # )

    # Example 2: CVAT → COCO
    # converter.cvat_to_coco(
    #     cvat_xml_path='annotations/annotations.xml',
    #     output_path='annotations/output.json',
    #     image_name_pattern='frame_{:06d}.png'
    # )

    # Example 3: COCO → YOLO
    # converter.coco_to_yolo(
    #     coco_json_path='annotations/H03.json',
    #     output_dir='data/labels/train',
    #     use_filename=True
    # )

    # Example 4: COCO → CVAT
    # converter.coco_to_cvat(
    #     coco_json_path='annotations/H03.json',
    #     output_path='annotations/output_cvat.xml'
    # )

    # Example 5: YOLO → COCO
    # converter.yolo_to_coco(
    #     labels_dir='data/labels/train',
    #     output_path='annotations/output.json',
    #     image_size=(1920, 1080),
    #     class_names={0: 'bird'}
    # )

    # Example 6: YOLO → CVAT
    # converter.yolo_to_cvat(
    #     labels_dir='data/labels/train',
    #     output_path='annotations/output_cvat.xml',
    #     image_size=(1920, 1080),
    #     class_names={0: 'bird'}
    # )

    # Example 7: Create YAML config
    # converter.create_yaml_config(
    #     output_path='data/dataset.yaml',
    #     dataset_path='.',
    #     train_dir='images/train',
    #     val_dir='images/val',
    #     test_dir='images/test'
    # )

    print("AnnotationConverter ready to use!")
    print("\nClass mapping:", converter.class_mapping)


if __name__ == '__main__':
    main()
