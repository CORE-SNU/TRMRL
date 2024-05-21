import numpy as np
import xml.etree.ElementTree as ET


def main():
    # TODO: use PyMJCF: see https://github.com/google-deepmind/dm_control/blob/main/tutorial.ipynb.
    N_TASKS = 125
    for task in range(N_TASKS):
        # TODO: retrieve the following values from the XML automatically
        th = .2 * (2. * np.random.rand() - 1.)
        floor_com = np.array([248., 0.])
        contact_point = np.array([0., .2])

        c = np.cos(-th)
        s = np.sin(-th)
        rotation_mat = np.array([[c, -s], [s, c]])
        transformed_pos = floor_com + rotation_mat @ (contact_point - floor_com) + np.array([0., 1.1])
        pos = np.array([transformed_pos[0], 0., transformed_pos[1]])
        transformed_zaxis = rotation_mat @ np.array([0., 1.])
        zaxis = np.array([transformed_zaxis[0], 0., transformed_zaxis[1]])

        tree = ET.parse('walker.xml')
        root = tree.getroot()

        worldbody = root.find('worldbody')
        worldbody_geom = worldbody.find('geom')

        default = root.find('default')

        default_friction = np.array([float(val_str) for val_str in default.find('geom').get('friction').split()])
        default_dampling = float(default.find('joint').get('damping'))

        worldbody_geom.set('zaxis', '{} {} {}'.format(*zaxis))
        torso = worldbody.find('body')
        torso.set('pos', '{} {} {}'.format(*pos))

        # perturb the links' mass & inertia, friction randomly
        base = 1.3
        log_scale_lim = 3.0

        for geom in torso.iter('geom'):
            exp_density = np.random.uniform(-log_scale_lim, log_scale_lim)
            exp_friction = np.random.uniform(-log_scale_lim, log_scale_lim, size=(3,))
            multiplier_density = base ** exp_density
            multiplier_friction = base ** exp_friction
            density = multiplier_density * 1000.  # 1000: default density of MuJoCo
            friction = multiplier_friction * default_friction
            geom.set('density', '{}'.format(density))               # mass & inertia
            geom.set('friction', '{} {} {}'.format(*friction))      # friction

            if geom.get('name') in ['left_foot', 'right_foot']:     # foot length
                size_rad, size_half_len = [float(val_str) for val_str in geom.get('size').split()]
                exp_size = np.random.uniform(-log_scale_lim/2, log_scale_lim)   # prevent the feet from being too small
                multiplier_size = base ** exp_size
                print('before', size_half_len)
                size_half_len *= multiplier_size
                print('after', size_half_len)
                geom.set('size', '{} {}'.format(size_rad, size_half_len))


        for joint in torso.iter('joint'):
            exp_damping = np.random.uniform(-log_scale_lim, log_scale_lim)
            multiplier_damping = base ** exp_damping
            damping = multiplier_damping * default_dampling
            joint.set('damping', '{}'.format(damping))              # damping

        tree.write('walker_assets/walker{}.xml'.format(task))


if __name__ == '__main__':
    main()