"""Launch elevation_mapping_cupy fed by DLIO's deskewed point cloud, with a
`quad` argument for multi-robot namespacing (matching the convention already
established across this workspace's acl-mapping/travel/DLIO launch files).

Usage (root namespace / single robot / bag playback):
  ros2 launch elevation_mapping_cupy elevation_mapping_dlio.launch.py

Usage (namespaced hardware run):
  ros2 launch elevation_mapping_cupy elevation_mapping_dlio.launch.py quad:=RR08

core_param.yaml and config/setups/dlio/base.yaml are loaded here as
flattened Python dicts (not raw file paths) rather than via `parameters=[file_path]`
directly, because core_param.yaml uses ROS2's absolute-node-name YAML key
(`/elevation_mapping_node:`), which ROS2 only applies to a node whose fully
qualified name matches EXACTLY -- it silently no-ops once this node is
namespaced (e.g. `/RR08/elevation_mapping_node`). A plain dict passed via
`Node(parameters=[...])` always applies regardless of namespace. This is the
same workaround already used in acl-mapping's global_mapper_travel.launch.py
for travel_ros's equally absolute-keyed kitti_params.yaml -- see that file's
comment for the original discovery (GroundGrid hit the identical issue with
mid360.yaml).

Parameter overlays (`param_overlays:=a.yaml,b.yaml`): deployment-owned YAML files applied in
order on top of core_param.yaml + dlio/base.yaml (and the plugin configs) -- e.g. a vehicle-type
file, then a single-vehicle file. Strict: an unknown key, a wrong type or a parameter this file
computes from its arguments stops the launch. See param_overlays.py next to this file for the
file format. `extra_cloud_topics:=t1,t2` (or an overlay's inputs.extra_clouds) fuses more
PointCloud2 topics alongside the DLIO cloud; `extra_cloud_exclude:=a,b*` refuses topics the DLIO
cloud already contains, so nothing is fused twice.
"""

import importlib.util
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _param_overlays_module():
    """param_overlays.py from this launch file's directory, loaded by path: it needs only PyYAML,
    and importing it through the elevation_mapping_cupy package would import cupy."""
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'param_overlays.py')
    spec = importlib.util.spec_from_file_location('elevation_mapping_cupy_param_overlays', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split(value: str) -> list:
    return [v.strip() for v in value.split(',') if v.strip()]


def _load_ros_parameters(yaml_path: str) -> dict:
    """Extract the ros__parameters dict regardless of whether the file uses
    the `/**:` wildcard key or an absolute `/<node_name>:` key -- both are
    valid ROS2 param-file top-level keys, and this launch file bypasses
    ROS2's own node-name matching entirely by flattening to a plain dict
    (see this file's header comment), so either convention works here."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    for key, value in data.items():
        if isinstance(value, dict) and 'ros__parameters' in value:
            return value['ros__parameters']
    raise ValueError(f"{yaml_path}: no '<key>: {{ros__parameters: ...}}' section found")


def launch_setup(context, *args, **kwargs):
    quad = LaunchConfiguration('quad').perform(context)
    deskewed_cloud_topic = LaunchConfiguration('deskewed_cloud_topic').perform(context)
    map_frame_id = LaunchConfiguration('map_frame_id').perform(context)
    base_frame_id = LaunchConfiguration('base_frame_id').perform(context)
    sensor_frame_id = LaunchConfiguration('sensor_frame_id').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)
    launch_rviz = LaunchConfiguration('launch_rviz')
    rviz_config = LaunchConfiguration('rviz_config').perform(context)

    share_dir = get_package_share_directory('elevation_mapping_cupy')
    core_param_path = os.path.join(share_dir, 'config', 'core', 'core_param.yaml')
    dlio_param_path = os.path.join(share_dir, 'config', 'setups', 'dlio', 'base.yaml')
    traversability_param_path = os.path.join(
        share_dir, 'config', 'core', 'traversability_plugin_config.yaml')

    plugin_param_path = os.path.join(share_dir, 'config', 'core', 'plugin_config.yaml')

    params = {}
    params.update(_load_ros_parameters(core_param_path))
    params.update(_load_ros_parameters(dlio_param_path))

    # Layer 3: deployment overlays (vehicle type, then vehicle), applied in the order given.
    po = _param_overlays_module()
    overlay_paths = _split(LaunchConfiguration('param_overlays').perform(context))
    base_extra_plugins = po.load_plugin_files([traversability_param_path])
    result = po.apply_overlays(
        params, po.load_plugin_files([plugin_param_path, traversability_param_path]), overlay_paths)
    params = result.node_params
    # Geometric traversability pipeline (normals/slope/roughness/step/fusion/occupancy/ESDF),
    # loaded as a second plugin file merged on top of plugin_config.yaml -- see
    # parameter.py:extra_plugin_config_file and elevation_mapping.py's load_plugin_settings
    # call. Each stage still has its own enable:true/false in that file; this just makes the
    # file itself available to load. With plugin overlays the node gets a generated copy
    # instead: this file's entries plus the full merged entry of every plugin an overlay changed.
    params['extra_plugin_config_file'] = traversability_param_path
    if result.plugin_overrides:
        fd, generated = tempfile.mkstemp(prefix=f"elevation_mapping_plugins_{quad or 'root'}_", suffix='.yaml')
        os.close(fd)
        params['extra_plugin_config_file'] = po.write_plugin_file(
            base_extra_plugins, result.plugin_overrides, generated)
    params['param_overlays_applied'] = result.applied_string()

    # Overrides -- see dlio/base.yaml's header comment for why these three
    # (topic/map_frame/base_frame) are placeholders in the file itself.
    params['subscribers'] = {
        'dlio_cloud': {
            'topic_name': deskewed_cloud_topic,
            'data_type': 'pointcloud',
        }
    }
    # Extra point clouds fused alongside the DLIO cloud (sensors DLIO does not ingest, e.g. a
    # depth camera). Each is moved into the map frame through TF at its own stamp and is NOT
    # motion-deskewed, so lidars belong in the merge before DLIO instead.
    extra_topics = _split(LaunchConfiguration('extra_cloud_topics').perform(context))
    extra_topics += [t for t in result.extra_clouds if t not in extra_topics]
    params['subscribers'].update(po.extra_cloud_subscribers(
        extra_topics, quad, deskewed_cloud_topic,
        _split(LaunchConfiguration('extra_cloud_exclude').perform(context))))
    params['map_frame'] = map_frame_id
    params['base_frame'] = base_frame_id
    params['corrected_map_frame'] = map_frame_id
    # Frame treated as the sensor origin for the noise model / validity gate / visibility
    # ray-casting when the deskewed cloud arrives already in the map frame -- see
    # elevation_mapping_node.py pointcloud_callback. Defaults to the lidar frame (DLIO's
    # frames/lidar); the node falls back to base_frame if this is empty/unavailable.
    params['sensor_frame'] = sensor_frame_id
    params['use_sim_time'] = use_sim_time.lower() in ('true', '1')
    # Output topic names for the three 2D grids. local_mapper_host_tmux.sh always passes
    # elev/* (occupancy_grid_topic:=elev/occ_2d_topic etc.) and map_fusion_node on the NUC
    # owns the canonical names MIGHTY reads. The defaults are those canonical names, kept
    # only so an Orin rebuilt before its local_mapper script is updated still feeds MIGHTY.
    params['occupancy_grid_topic'] = LaunchConfiguration('occupancy_grid_topic').perform(context)
    params['planning_occupancy_grid_topic'] = LaunchConfiguration('planning_occupancy_grid_topic').perform(context)
    params['esdf_grid_topic'] = LaunchConfiguration('esdf_grid_topic').perform(context)

    elevation_mapping_node = Node(
        package='elevation_mapping_cupy',
        executable='elevation_mapping_node.py',
        name='elevation_mapping_node',
        namespace=quad if quad else None,
        output='screen',
        parameters=[params],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        namespace=quad if quad else None,
        arguments=['-d', rviz_config] if rviz_config else [],
        parameters=[{'use_sim_time': use_sim_time.lower() in ('true', '1')}],
        output='screen',
        condition=IfCondition(launch_rviz),
    )

    log = [LogInfo(msg=line) for line in po.describe(result)]
    for key, sub in params['subscribers'].items():
        if key != 'dlio_cloud':
            log.append(LogInfo(msg=f"extra cloud input {key}: {sub['topic_name']}"))
    if result.plugin_overrides:
        log.append(LogInfo(msg=f"plugin overlays -> {params['extra_plugin_config_file']}"))
    return log + [elevation_mapping_node, rviz_node]


def generate_launch_description() -> LaunchDescription:
    # Mirrors dlio/odom.cc's own frame-namespacing (get_namespace(), leading
    # slash stripped, "<ns>/odom" / "<ns>/base_link") and this workspace's
    # other DLIO-consumer launch files (dlio_deskewed_elevation.launch.py) --
    # empty quad = root namespace, matching DLIO's own "skip prefixing
    # entirely at the root namespace" behavior.
    quad_arg = DeclareLaunchArgument(
        'quad', default_value='',
        description='Robot namespace DLIO is launched under, e.g. RR08. '
                    'Empty (default) = root namespace / bag playback.',
    )

    quad = LaunchConfiguration('quad')
    default_deskewed_topic = PythonExpression(
        ["'/' + '", quad, "' + '/dlio/odom_node/pointcloud/deskewed' if '", quad, "' else 'dlio/odom_node/pointcloud/deskewed'"])
    default_map_frame_id = PythonExpression(["'", quad, "/odom' if '", quad, "' else 'odom'"])
    default_base_frame_id = PythonExpression(["'", quad, "/base_link' if '", quad, "' else 'base_link'"])
    # DLIO's frames/lidar (see its cfg/params.yaml), namespaced the same way as odom/base_link.
    default_sensor_frame_id = PythonExpression(["'", quad, "/lidar' if '", quad, "' else 'lidar'"])

    deskewed_topic_arg = DeclareLaunchArgument(
        'deskewed_cloud_topic', default_value=default_deskewed_topic,
        description='DLIO deskewed PointCloud2 topic (dlio/odom.cc: deskewed_pub).',
    )
    map_frame_arg = DeclareLaunchArgument(
        'map_frame_id', default_value=default_map_frame_id,
        description="Must match DLIO's this->odom_frame exactly.",
    )
    base_frame_arg = DeclareLaunchArgument(
        'base_frame_id', default_value=default_base_frame_id,
        description="Must match DLIO's this->baselink_frame exactly.",
    )
    sensor_frame_arg = DeclareLaunchArgument(
        'sensor_frame_id', default_value=default_sensor_frame_id,
        description="Lidar frame used as the sensor origin for the noise model / height gate "
                    "/ visibility ray-casting (DLIO's frames/lidar). Node falls back to "
                    "base_frame if empty or its TF is unavailable.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation clock if true (e.g. bag playback with --clock).',
    )
    launch_rviz_arg = DeclareLaunchArgument(
        'launch_rviz', default_value='false', description='Whether to launch RViz.',
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value='', description='Path to an RViz config file.',
    )

    occupancy_grid_topic_arg = DeclareLaunchArgument(
        'occupancy_grid_topic', default_value='occ_2d_topic',
        description="Tri-state occupancy output, relative to the namespace. local_mapper always passes "
                    "elev/occ_2d_topic; the default (MIGHTY's name) exists only so an Orin "
                    "rebuilt before its local_mapper script is updated still feeds MIGHTY.",
    )
    planning_occupancy_grid_topic_arg = DeclareLaunchArgument(
        'planning_occupancy_grid_topic', default_value='planning_occ_2d_topic',
        description="Planning occupancy output. local_mapper always passes elev/planning_occ_2d_topic; "
                    "the default (MIGHTY's name) exists only so an Orin rebuilt before its "
                    "local_mapper script is updated still feeds MIGHTY.",
    )
    esdf_grid_topic_arg = DeclareLaunchArgument(
        'esdf_grid_topic', default_value='esdf_2d_topic',
        description="ESDF output. local_mapper always passes elev/esdf_2d_topic; the default "
                    "(MIGHTY's name) exists only so an Orin rebuilt before its local_mapper "
                    "script is updated still feeds MIGHTY.",
    )

    param_overlays_arg = DeclareLaunchArgument(
        'param_overlays', default_value='',
        description='Comma-separated parameter overlay YAML files applied in order over '
                    'core_param.yaml + dlio/base.yaml (see param_overlays.py). Empty = none.',
    )
    extra_cloud_topics_arg = DeclareLaunchArgument(
        'extra_cloud_topics', default_value='',
        description='Comma-separated extra PointCloud2 topics fused alongside the DLIO cloud '
                    '(added to any overlay inputs.extra_clouds). Not deskewed: for sensors DLIO '
                    'does not ingest.',
    )
    extra_cloud_exclude_arg = DeclareLaunchArgument(
        'extra_cloud_exclude', default_value='',
        description="Comma-separated topics the DLIO cloud already contains; an entry ending in "
                    "'*' is a prefix. Listing one as an extra cloud stops the launch.",
    )

    return LaunchDescription([
        quad_arg,
        param_overlays_arg,
        extra_cloud_topics_arg,
        extra_cloud_exclude_arg,
        occupancy_grid_topic_arg,
        planning_occupancy_grid_topic_arg,
        esdf_grid_topic_arg,
        deskewed_topic_arg,
        map_frame_arg,
        base_frame_arg,
        sensor_frame_arg,
        use_sim_time_arg,
        launch_rviz_arg,
        rviz_config_arg,
        OpaqueFunction(function=launch_setup),
    ])
