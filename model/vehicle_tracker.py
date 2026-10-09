"""Reusable centroid tracker for sampled vehicle detections."""

from __future__ import annotations

from dataclasses import dataclass
import math

from model.vehicle_detector import VehicleDetection


@dataclass
class Track:
    track_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    previous_center: tuple[float, float]
    confidence: float
    missed_frames: int = 0
    distance: float = 0.0


class CentroidTracker:
    """Match detections to persistent IDs using class-aware centroid distance."""

    def __init__(self, max_distance: float = 80.0, max_missed_frames: int = 10) -> None:
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self._next_id = 1
        self.tracks: dict[int, Track] = {}

    @staticmethod
    def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, detections: list[VehicleDetection]) -> list[Track]:
        candidates = [
            (self._center(detection["bbox"]), index, detection)
            for index, detection in enumerate(detections)
        ]
        assignments: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for center, detection_index, detection in candidates:
                if detection["class_name"] == track.class_name:
                    distance = math.dist(track.center, center)
                    if distance <= self.max_distance:
                        assignments.append((distance, track_id, detection_index))
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_id, detection_index in sorted(assignments):
            if track_id in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(detection_index)
            detection = detections[detection_index]
            track = self.tracks[track_id]
            center = self._center(detection["bbox"])
            track.previous_center = track.center
            track.center = center
            track.bbox = detection["bbox"]
            track.confidence = detection["confidence"]
            track.distance = math.dist(track.previous_center, center)
            track.missed_frames = 0

        for track_id, track in list(self.tracks.items()):
            if track_id not in used_tracks:
                track.missed_frames += 1
                track.distance = 0.0
                if track.missed_frames > self.max_missed_frames:
                    del self.tracks[track_id]

        for detection_index, detection in enumerate(detections):
            if detection_index in used_detections:
                continue
            center = self._center(detection["bbox"])
            self.tracks[self._next_id] = Track(
                track_id=self._next_id,
                class_name=detection["class_name"],
                bbox=detection["bbox"],
                center=center,
                previous_center=center,
                confidence=detection["confidence"],
            )
            self._next_id += 1

        return [self.tracks[track_id] for track_id in sorted(used_tracks)] + [
            self.tracks[track_id]
            for track_id in sorted(self.tracks)
            if track_id not in used_tracks and self.tracks[track_id].missed_frames == 0
        ]
