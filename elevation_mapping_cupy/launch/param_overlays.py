"""Layered parameter overlays for elevation_mapping_dlio.launch.py.

Parameters are resolved in layers, each overriding the ones before it:

  1. config/core/core_param.yaml (+ plugin_config.yaml)         upstream defaults
  2. config/setups/dlio/base.yaml (+ traversability_plugin_config.yaml)   how a DLIO rover runs this
  3. overlay files from the `param_overlays` launch argument, in order (e.g. a vehicle-type
     file, then a single-vehicle file; the files live with the deployment, not in this repo)
  4. values the launch file computes from its own arguments (frames, topics, subscribers)

An overlay is a YAML mapping with up to three sections:

  node:      elevation_mapping_node ROS parameters, deep-merged over layers 1-2
  plugins:   plugin settings, deep-merged per plugin over plugin_config.yaml +
             traversability_plugin_config.yaml
  inputs:
    extra_clouds: [topic, ...]   extra PointCloud2 subscribers fused alongside the main cloud

Overlays are strict. A key that the base layers do not define, a value of the wrong type, a
parameter the launch file computes itself, or an unknown section stops the launch with the
file and key named. A typo can never silently do nothing. To make a new parameter
overridable, add it to a base file first.

This module needs only PyYAML: the launch file loads it by path, so importing it never imports
the elevation_mapping_cupy package (whose __init__ pulls in cupy).
"""

import copy
import hashlib
import os

import yaml

SECTIONS = ('node', 'plugins', 'inputs')
INPUT_KEYS = ('extra_clouds',)

# Node parameters the launch file sets from its own arguments after the overlays are applied.
# An overlay value for them would be overwritten, so it is refused instead.
LAUNCH_OWNED = frozenset({
    'subscribers', 'map_frame', 'base_frame', 'corrected_map_frame', 'sensor_frame',
    'use_sim_time', 'occupancy_grid_topic', 'planning_occupancy_grid_topic', 'esdf_grid_topic',
    'extra_plugin_config_file', 'param_overlays_applied',
})


class OverlayError(ValueError):
    """An overlay file that must not be applied (bad section, key, type or input)."""


def load_ros_parameters(yaml_path):
    """The ros__parameters dict of a ROS 2 params file, whatever its node-name key."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    for value in (data or {}).values():
        if isinstance(value, dict) and 'ros__parameters' in value:
            return value['ros__parameters']
    raise ValueError(f"{yaml_path}: no '<key>: {{ros__parameters: ...}}' section found")


def load_plugin_files(paths):
    """Plugin config as the node builds it: later files replace earlier entries by plugin name."""
    cfg = {}
    for path in paths:
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg


def file_digest(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def load_overlay(path):
    if not os.path.isfile(path):
        raise OverlayError(f"{path}: overlay file not found")
    with open(path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise OverlayError(f"{path}: not valid YAML: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise OverlayError(f"{path}: an overlay must be a mapping with sections {', '.join(SECTIONS)}")
    unknown = sorted(set(data) - set(SECTIONS))
    if unknown:
        raise OverlayError(f"{path}: unknown section(s) {unknown}; allowed: {', '.join(SECTIONS)}")
    for section in ('node', 'plugins', 'inputs'):
        if data.get(section) is not None and not isinstance(data[section], dict):
            raise OverlayError(f"{path}: section '{section}' must be a mapping")
    return data


def _coerce(base_value, value, where):
    """The overlay value, typed like the base value. int -> float is allowed (YAML writes 1 for 1.0,
    and a float ROS parameter given an int arrives as an INTEGER parameter); nothing else changes type."""
    if base_value is None:
        return value
    if isinstance(base_value, bool) or isinstance(value, bool):
        if isinstance(base_value, bool) and isinstance(value, bool):
            return value
    elif isinstance(base_value, float) and isinstance(value, (int, float)):
        return float(value)
    elif isinstance(base_value, int) and isinstance(value, int):
        return value
    elif isinstance(base_value, str) and isinstance(value, str):
        return value
    elif isinstance(base_value, list) and isinstance(value, list):
        return list(value)
    raise OverlayError(f"{where}: {type(value).__name__} {value!r} does not match the base "
                       f"{type(base_value).__name__} {base_value!r}")


def _merge(base, over, path, prefix, changes, source):
    for key, value in over.items():
        dotted = f"{prefix}{key}"
        where = f"{path}: '{dotted}'"
        if key not in base:
            raise OverlayError(f"{where} is not defined by the base parameters; check the spelling, "
                               "or add it to a base file first")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise OverlayError(f"{where} is a group in the base parameters; give a mapping")
            _merge(base[key], value, path, dotted + '.', changes, source)
            continue
        if isinstance(value, dict):
            raise OverlayError(f"{where} is a single value in the base parameters, not a group")
        new = _coerce(base[key], value, where)
        if new != base[key]:
            changes.append((source, dotted, base[key], new))
        base[key] = new


class OverlayResult:
    def __init__(self, node_params, plugin_overrides, extra_clouds, applied, changes):
        self.node_params = node_params            # layers 1-3 node parameters (deep copy)
        self.plugin_overrides = plugin_overrides  # {plugin name: full merged entry} for touched plugins
        self.extra_clouds = extra_clouds          # topics from inputs.extra_clouds, in order, deduplicated
        self.applied = applied                    # [(path, sha256[:12])]
        self.changes = changes                    # [(path, dotted key, old, new)]

    def applied_string(self):
        return ','.join(f"{p}@{d}" for p, d in self.applied) or 'none'


def apply_overlays(node_params, plugin_cfg, overlay_paths):
    """Apply overlay files in order. Inputs are not modified."""
    node = copy.deepcopy(node_params)
    plugins = copy.deepcopy(plugin_cfg)
    touched, clouds, applied, changes = [], [], [], []
    for path in overlay_paths:
        data = load_overlay(path)
        for key in sorted(set(data.get('node') or {}) & LAUNCH_OWNED):
            raise OverlayError(f"{path}: 'node.{key}' is set by the launch file from its arguments "
                               "(frames, topics, subscribers); pass it as a launch argument instead")
        _merge(node, data.get('node') or {}, path, 'node.', changes, path)
        for name, entry in (data.get('plugins') or {}).items():
            if name not in plugins:
                raise OverlayError(f"{path}: 'plugins.{name}' is not a plugin in plugin_config.yaml "
                                   "or traversability_plugin_config.yaml")
            if not isinstance(entry, dict):
                raise OverlayError(f"{path}: 'plugins.{name}' must be a mapping")
            _merge(plugins[name], entry, path, f'plugins.{name}.', changes, path)
            if name not in touched:
                touched.append(name)
        inputs = data.get('inputs') or {}
        unknown = sorted(set(inputs) - set(INPUT_KEYS))
        if unknown:
            raise OverlayError(f"{path}: unknown input(s) {unknown}; allowed: {', '.join(INPUT_KEYS)}")
        extra = inputs.get('extra_clouds') or []
        if not isinstance(extra, list) or not all(isinstance(t, str) and t.strip() for t in extra):
            raise OverlayError(f"{path}: 'inputs.extra_clouds' must be a list of topic names")
        for topic in extra:
            if topic.strip() not in clouds:
                clouds.append(topic.strip())
        applied.append((path, file_digest(path)))
    return OverlayResult(node, {n: plugins[n] for n in touched}, clouds, applied, changes)


def resolve_topic(topic, namespace):
    """Absolute topic name as rclpy resolves `topic` for a node in `namespace`."""
    topic = topic.strip()
    if topic.startswith('/'):
        return topic
    return f"/{namespace.strip('/')}/{topic}" if namespace.strip('/') else f"/{topic}"


def extra_cloud_subscribers(topics, namespace, main_cloud_topic, exclude=()):
    """Subscriber entries for extra point clouds, refusing any that would fuse a cloud twice.

    `exclude` lists topics the main cloud already contains (e.g. lidars merged before DLIO); an
    entry ending in '*' excludes every topic with that prefix. Names may be relative to the
    node's namespace or absolute."""
    main = resolve_topic(main_cloud_topic, namespace)
    rules = [(e.strip()[:-1], True) if e.strip().endswith('*') else (e.strip(), False)
             for e in exclude if e.strip()]
    rules = [(resolve_topic(p, namespace) if p else '/', prefix) for p, prefix in rules]
    subs, seen = {}, []
    for topic in topics:
        full = resolve_topic(topic, namespace)
        if full == main:
            raise OverlayError(f"extra cloud {topic!r} is the main cloud ({main}); it would be fused twice")
        for pattern, prefix in rules:
            if (prefix and full.startswith(pattern)) or (not prefix and full == pattern):
                raise OverlayError(f"extra cloud {topic!r} ({full}) is excluded by "
                                   f"{pattern + ('*' if prefix else '')!r}: the main cloud already "
                                   "contains it, so it would be fused twice")
        if full in seen:
            continue
        seen.append(full)
        subs[f"extra_cloud_{len(subs)}"] = {'topic_name': topic.strip(), 'data_type': 'pointcloud'}
    return subs


def write_plugin_file(base_extra_cfg, overrides, path):
    """Plugin file loaded after plugin_config.yaml: the base extra file's entries plus the full
    merged entry of every plugin an overlay touched (the node replaces entries by plugin name)."""
    cfg = copy.deepcopy(base_extra_cfg)
    cfg.update(copy.deepcopy(overrides))
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def describe(result):
    """Human-readable lines for the launch log."""
    if not result.applied:
        return ['param overlays: none']
    lines = ['param overlays (applied in order):']
    lines += [f"  {p} (sha256 {d})" for p, d in result.applied]
    for path, key, old, new in result.changes:
        lines.append(f"  {key}: {old!r} -> {new!r}   [{os.path.basename(path)}]")
    if not result.changes:
        lines.append('  (no value differs from the base parameters)')
    return lines
