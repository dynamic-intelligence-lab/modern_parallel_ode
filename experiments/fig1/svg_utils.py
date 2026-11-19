"""SVG helpers of make_figure.py: write the .svg, move the definitions to the root, group each merit panel."""
import xml.etree.ElementTree as ET
SVG='http://www.w3.org/2000/svg'
XLINK='http://www.w3.org/1999/xlink'
ET.register_namespace('',SVG)
ET.register_namespace('xlink',XLINK)
def tag(name):
    return f'{{{SVG}}}{name}'

def write_svg(root, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)

def normalize_svg(root):
    """Keep SVG 1.1 text, with reusable definitions at root."""
    parents = {child: parent for parent in root.iter() for child in parent}
    definitions = ET.Element(tag('defs'))
    for old in list(root.iter(tag('defs'))):
        definitions.extend(list(old))
        parents[old].remove(old)
    root.insert(0, definitions)
    # Axes3D draws its frame before opening the ordinary Axes SVG group.
    # Reunite these siblings so moving/extracting a merit panel retains axes.
    for parent in list(root.iter()):
        for panel in list(parent):
            name = panel.get('id', '')
            if not name.endswith('_merit'):
                continue
            siblings = list(parent)
            index = siblings.index(panel)
            start = index
            while start > 0:
                previous_id = siblings[start-1].get('id', '')
                if previous_id == 'patch_1' or not previous_id.startswith(
                        ('pane3d_', 'grid3d_', 'axis3d_', 'patch_')):
                    break
                start -= 1
            if start < index:
                wrapper = ET.Element(tag('g'), {'id': name})
                panel.set('id', name + '_artwork')
                for item in siblings[start:index+1]:
                    parent.remove(item)
                    wrapper.append(item)
                parent.insert(start, wrapper)
    for node in root.iter():
        gid = node.get('id', '')
        if '_surface_' in gid:
            # Adjacent filled facets must not acquire artificial white seams.
            node.set('shape-rendering', 'crispEdges')
        if node.tag in (tag('text'), tag('tspan')):
            # Explicit SVG font attributes are portable across vector editors.
            remaining = []
            for declaration in node.get('style', '').split(';'):
                key, sep, value = declaration.strip().partition(':')
                if not sep:
                    continue
                if key.startswith('font-') or key == 'text-anchor':
                    node.set(key, value.strip())
                else:
                    remaining.append(declaration.strip())
            if remaining:
                node.set('style', '; '.join(remaining))
            else:
                node.attrib.pop('style', None)
    return root
