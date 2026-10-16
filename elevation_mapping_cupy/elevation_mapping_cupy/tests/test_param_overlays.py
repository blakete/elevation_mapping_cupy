"""Parameter overlays (launch/param_overlays.py) and their wiring in elevation_mapping_dlio.launch.py.

The module tests need only PyYAML. The launch tests run launch_setup against this package's own
config files and are skipped where ROS 2 `launch` / `launch_ros` are not installed. Neither needs a
GPU: the overlay module is loaded by path, so the cupy-importing package is never imported.
"""
import importlib.util
import os
import textwrap

import pytest
import yaml

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LAUNCH_DIR = os.path.join(PKG, 'launch')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


po = _load('param_overlays_under_test', os.path.join(LAUNCH_DIR, 'param_overlays.py'))

BASE = {'min_valid_distance': 0.0, 'map_length': 15.0, 'pointcloud_qos_depth': 1, 'enable_x': True,
        'name': 'a', 'group': {'inner': 2.0, 'other': 'keep'}, 'layers': ['elevation']}
PLUGINS = {'occupancy': {'enable': True, 'layer_name': 'occupancy',
                         'extra_params': {'hazard_maximum_variance': 5.0, 'lethal': 0.35}},
           'min_filter': {'enable': True, 'extra_params': {'iteration_n': 30}}}


@pytest.fixture
def overlay(tmp_path):
    def make(text, name='o.yaml'):
        p = tmp_path / name
        p.write_text(textwrap.dedent(text))
        return str(p)
    return make


def test_no_overlays_changes_nothing():
    r = po.apply_overlays(BASE, PLUGINS, [])
    assert r.node_params == BASE and r.node_params is not BASE
    assert r.plugin_overrides == {} and r.extra_clouds == [] and r.changes == []
    assert r.applied_string() == 'none'
    assert po.describe(r) == ['param overlays: none']


def test_layers_apply_in_order_and_later_wins(overlay):
    jk = overlay('node: {min_valid_distance: 0.4, map_length: 20}\n', 'JK.yaml')
    one = overlay('node: {map_length: 25.0}\n', 'JK01.yaml')
    r = po.apply_overlays(BASE, PLUGINS, [jk, one])
    assert r.node_params['min_valid_distance'] == 0.4
    assert r.node_params['map_length'] == 25.0
    assert BASE['map_length'] == 15.0  # input untouched
    assert [(os.path.basename(p), k) for p, k, _, _ in r.changes] == [
        ('JK.yaml', 'node.min_valid_distance'), ('JK.yaml', 'node.map_length'), ('JK01.yaml', 'node.map_length')]
    assert [os.path.basename(p) for p, _ in r.applied] == ['JK.yaml', 'JK01.yaml']
    assert all(len(d) == 12 for _, d in r.applied)


def test_int_for_float_is_coerced_to_float(overlay):
    r = po.apply_overlays(BASE, PLUGINS, [overlay('node: {map_length: 20}\n')])
    assert r.node_params['map_length'] == 20.0 and isinstance(r.node_params['map_length'], float)


def test_nested_group_merge_keeps_siblings(overlay):
    r = po.apply_overlays(BASE, PLUGINS, [overlay('node: {group: {inner: 3}}\n')])
    assert r.node_params['group'] == {'inner': 3.0, 'other': 'keep'}


@pytest.mark.parametrize('text, needle', [
    ('node: {min_valid_distanse: 0.4}\n', "'node.min_valid_distanse' is not defined"),
    ('node: {map_length: "15"}\n', 'does not match the base float'),
    ('node: {pointcloud_qos_depth: 1.5}\n', 'does not match the base int'),
    ('node: {enable_x: 1}\n', 'does not match the base bool'),
    ('node: {group: 3}\n', 'is a group in the base parameters'),
    ('node: {name: {x: 1}}\n', 'is a single value'),
    ('node: {map_frame: odom}\n', 'set by the launch file'),
    ('node: {subscribers: {x: {topic_name: t}}}\n', 'set by the launch file'),
    ('nodes: {map_length: 3.0}\n', 'unknown section'),
    ('node: [1, 2]\n', "section 'node' must be a mapping"),
    ('- a\n', 'must be a mapping'),
    ('plugins: {nope: {enable: false}}\n', "'plugins.nope' is not a plugin"),
    ('plugins: {occupancy: {extra_params: {lethall: 0.4}}}\n', "'plugins.occupancy.extra_params.lethall'"),
    ('inputs: {clouds: [a]}\n', 'unknown input'),
    ('inputs: {extra_clouds: a}\n', 'must be a list of topic names'),
])
def test_strict_errors_name_the_file_and_key(overlay, text, needle):
    path = overlay(text)
    with pytest.raises(po.OverlayError) as e:
        po.apply_overlays(BASE, PLUGINS, [path])
    assert needle in str(e.value)
    if 'section' not in needle and 'mapping' not in needle:
        assert path in str(e.value)


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(po.OverlayError, match='not found'):
        po.apply_overlays(BASE, PLUGINS, [str(tmp_path / 'absent.yaml')])


def test_empty_overlay_is_applied_and_recorded(overlay):
    r = po.apply_overlays(BASE, PLUGINS, [overlay('', 'empty.yaml')])
    assert r.node_params == BASE and len(r.applied) == 1
    assert 'no value differs' in '\n'.join(po.describe(r))


def test_plugin_overlay_returns_full_entries(overlay, tmp_path):
    r = po.apply_overlays(BASE, PLUGINS, [overlay('plugins: {occupancy: {extra_params: {lethal: 0.4}}}\n')])
    assert list(r.plugin_overrides) == ['occupancy']
    entry = r.plugin_overrides['occupancy']
    assert entry['extra_params'] == {'hazard_maximum_variance': 5.0, 'lethal': 0.4}
    assert entry['enable'] is True and entry['layer_name'] == 'occupancy'
    assert PLUGINS['occupancy']['extra_params']['lethal'] == 0.35
    out = po.write_plugin_file({'esdf': {'enable': True}}, r.plugin_overrides, str(tmp_path / 'p.yaml'))
    assert yaml.safe_load(open(out)) == {'esdf': {'enable': True}, 'occupancy': entry}


def test_extra_clouds_from_overlays_are_deduplicated_in_order(overlay):
    a = overlay('inputs: {extra_clouds: [d455/depth/points, other]}\n', 'a.yaml')
    b = overlay('inputs: {extra_clouds: [other, third]}\n', 'b.yaml')
    assert po.apply_overlays(BASE, PLUGINS, [a, b]).extra_clouds == ['d455/depth/points', 'other', 'third']


def test_resolve_topic():
    assert po.resolve_topic('livox/lidar', 'JK01') == '/JK01/livox/lidar'
    assert po.resolve_topic('/JK01/livox/lidar', 'JK01') == '/JK01/livox/lidar'
    assert po.resolve_topic('livox/lidar', '') == '/livox/lidar'


def test_extra_cloud_subscribers_shape_and_dedup():
    subs = po.extra_cloud_subscribers(['d455/depth/points', '/JK01/d455/depth/points', 'cam2/points'],
                                      'JK01', '/JK01/dlio/odom_node/pointcloud/deskewed')
    assert subs == {'extra_cloud_0': {'topic_name': 'd455/depth/points', 'data_type': 'pointcloud'},
                    'extra_cloud_1': {'topic_name': 'cam2/points', 'data_type': 'pointcloud'}}


def test_extra_cloud_refuses_the_main_cloud():
    with pytest.raises(po.OverlayError, match='main cloud'):
        po.extra_cloud_subscribers(['dlio/odom_node/pointcloud/deskewed'], 'JK01',
                                   '/JK01/dlio/odom_node/pointcloud/deskewed')


@pytest.mark.parametrize('topic', ['livox/lidar_192_168_1_140', '/JK01/livox/lidar_merged'])
def test_extra_cloud_refuses_excluded_prefix(topic):
    with pytest.raises(po.OverlayError, match='fused twice'):
        po.extra_cloud_subscribers([topic], 'JK01', 'dlio/odom_node/pointcloud/deskewed', ['livox/*'])


def test_extra_cloud_exact_exclude_only_matches_exactly():
    subs = po.extra_cloud_subscribers(['livox/lidar_2'], 'JK01', 'x', ['livox/lidar'])
    assert list(subs) == ['extra_cloud_0']
    with pytest.raises(po.OverlayError):
        po.extra_cloud_subscribers(['/JK01/livox/lidar'], 'JK01', 'x', ['livox/lidar'])


def test_real_base_files_accept_a_jackal_style_overlay(overlay):
    """The keys the fleet overlays use exist in this package's base files."""
    node = {}
    node.update(po.load_ros_parameters(os.path.join(PKG, 'config', 'core', 'core_param.yaml')))
    node.update(po.load_ros_parameters(os.path.join(PKG, 'config', 'setups', 'dlio', 'base.yaml')))
    plugins = po.load_plugin_files([os.path.join(PKG, 'config', 'core', 'plugin_config.yaml'),
                                    os.path.join(PKG, 'config', 'core', 'traversability_plugin_config.yaml')])
    assert node['pointcloud_qos_depth'] == 1
    r = po.apply_overlays(node, plugins, [overlay('node: {min_valid_distance: 0.4}\n')])
    assert r.node_params['min_valid_distance'] == 0.4


# ---- the launch file -------------------------------------------------------------------------

def _launch_setup(monkeypatch, **args):
    pytest.importorskip('launch')
    pytest.importorskip('launch_ros')
    from launch import LaunchContext
    lf = _load('eme_dlio_launch_under_test', os.path.join(LAUNCH_DIR, 'elevation_mapping_dlio.launch.py'))
    monkeypatch.setattr(lf, 'get_package_share_directory', lambda name: PKG)
    ld = lf.generate_launch_description()
    context = LaunchContext()
    for entity in ld.entities:
        if type(entity).__name__ == 'DeclareLaunchArgument':
            value = args.get(entity.name)
            if value is None:
                value = ''.join(s.perform(context) for s in entity.default_value) if entity.default_value else ''
            context.launch_configurations[entity.name] = value
    actions = lf.launch_setup(context)
    nodes = [a for a in actions if type(a).__name__ == 'Node']
    logs = [a for a in actions if type(a).__name__ == 'LogInfo']
    params = nodes[0]._Node__parameters[0]
    msgs = [''.join(getattr(s, 'text', '') for s in log.msg) for log in logs]
    return params, msgs


def _text(value):
    """launch_ros stores names and string values as tuples of TextSubstitution holding YAML
    text (it decodes them when the node starts); join and decode them the same way."""
    if isinstance(value, (tuple, list)) and value and all(hasattr(s, 'text') for s in value):
        text = ''.join(s.text for s in value)
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError:
            return text
    return value


def _params_dict(params):
    """Node(parameters=[dict]) after launch_ros normalization, as plain {name: value}."""
    return {_text(k): _text(v) for k, v in params.items()}


def test_launch_without_overlays_keeps_base(monkeypatch):
    params, msgs = _launch_setup(monkeypatch, quad='JK01')
    p = _params_dict(params)
    assert p['param_overlays_applied'] == 'none'
    assert p['pointcloud_qos_depth'] == 1
    assert p['extra_plugin_config_file'].endswith('traversability_plugin_config.yaml')
    assert 'param overlays: none' in msgs


def test_launch_applies_overlays_and_extra_clouds(monkeypatch, overlay):
    jk = overlay('node: {min_valid_distance: 0.4}\n'
                 'plugins: {occupancy: {extra_params: {lethal_traversability_threshold: 0.4}}}\n'
                 'inputs: {extra_clouds: [d455/depth/points]}\n', 'JK.yaml')
    params, msgs = _launch_setup(monkeypatch, quad='JK01', param_overlays=jk, extra_cloud_exclude='livox/*')
    p = _params_dict(params)
    assert p['min_valid_distance'] == 0.4
    assert p['param_overlays_applied'].startswith(jk + '@')
    plugin_file = p['extra_plugin_config_file']
    assert os.path.basename(plugin_file).startswith('elevation_mapping_plugins_JK01_')
    assert yaml.safe_load(open(plugin_file))['occupancy']['extra_params']['lethal_traversability_threshold'] == 0.4
    os.unlink(plugin_file)
    flat = {k: v for k, v in p.items() if k.startswith('subscribers')}
    assert any('extra_cloud_0' in str(k) for k in flat) or 'extra_cloud_0' in str(p.get('subscribers'))
    assert any('node.min_valid_distance: 0.0 -> 0.4' in m for m in msgs)
    assert any('extra cloud input extra_cloud_0: d455/depth/points' in m for m in msgs)


def test_launch_refuses_a_merged_lidar_as_extra_cloud(monkeypatch):
    with pytest.raises(Exception, match='fused twice'):
        _launch_setup(monkeypatch, quad='JK01', extra_cloud_topics='livox/lidar_192_168_1_140',
                      extra_cloud_exclude='livox/*')
