import bolt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from msk_envs.recording.frame_parser import FrameData
from msk_envs.geometry import FWD_IDX, UP_IDX, SIDE_IDX
from .plots import SequencePlot, PlotConfig


def create_generic_plot(
        names: list[str],
        times: np.ndarray,
        frame_ind: np.ndarray,
        plot_data: np.ndarray,
        fig_title: str,
        y_label: str,
        y_fmt: str,
        pdf: PdfPages,
        enforced_y_range: list[tuple[float, float]] = None,
        sublabels: list[str] = None,
        alphas: list[float] = None,
        linestyles: list[str] = None,
        horizontal_lines: list[list[float]] = None,
        omit_zeros: bool = False,
        omit_tolerance: float = 1e-4,
        secondary_overlay_data: np.ndarray = None,
):
    n_vertical, n_horizontal = 3, 1
    n_plots = plot_data.shape[1]
    figs_per_page = n_vertical * n_horizontal
    num_pages = (n_plots + figs_per_page - 1) // figs_per_page

    for idx_page in range(num_pages):
        seq_plot = SequencePlot(
            PlotConfig(
                num_vertical=n_vertical,
                num_horizontal=n_horizontal,
                fig_size=(8.5, 11),
                title=fig_title,
                x_label="Time (s)",
                x_label_sub="Frame",
                y_label=y_label,
                x_data=times,
                x_data_sub=frame_ind,
                x_fmt=".2f",
                x_sub_fmt=".0f",
                y_fmt=y_fmt
            )
        )

        # Add plots for this page
        for idx_fig in range(figs_per_page):
            # Retrieve data subset for this figure
            start_idx = (idx_page * figs_per_page + idx_fig)
            end_idx = start_idx + 1
            if start_idx >= n_plots:
                continue
            data_subset = plot_data[:, start_idx:end_idx]
            data_subset_names = names[start_idx:end_idx]
            title = ", ".join(data_subset_names)

            # Add each entry in the subset
            for i in range(data_subset.shape[1]):
                entry_name = data_subset_names[i]
                data_sequence = data_subset[:, i]

                if len(data_sequence.shape) == 1:  # simple 1d plot
                    seq_plot.add(idx_fig, data_sequence, label=entry_name, title=title)
                else:  # multiple values per entry (e.g., value and reference)
                    assert sublabels is not None
                    label = sublabels

                    for part in range(data_subset.shape[-1]):
                        alpha = 1.0 if alphas is None else alphas[part]
                        linestyle = None if linestyles is None else linestyles[part]
                        if omit_zeros and np.all(np.abs(data_subset[..., part]) < omit_tolerance):
                            continue
                        seq_plot.add(idx_fig, data_subset[..., part],
                                     label=label[part],
                                     alpha=alpha,
                                     linestyle=linestyle,
                                     title=title)

                # Add horizontal lines if specified. can be used for zero lines or joint limits
                idx_entry = start_idx + i
                if horizontal_lines and horizontal_lines[idx_entry] is not None:
                    for hline in horizontal_lines[idx_entry]:
                        if np.isfinite(hline):  # we use -inf, inf for limits that are not defined
                            seq_plot.add_hline(idx_fig, hline)

                if enforced_y_range is not None and enforced_y_range[start_idx] is not None:
                    seq_plot.enforce_y_range(idx_fig, enforced_y_range[start_idx][0], enforced_y_range[start_idx][1])

                # If overlayed data requested
                if secondary_overlay_data is not None:
                    seq_plot.add_overlay(idx_fig, secondary_overlay_data[start_idx])

        seq_plot.finish(pdf)


def _create_interval_plots(interval_duration: float, times: np.ndarray, fn):
    time_current, final_time = 0.0, times[-1]
    if final_time - time_current > interval_duration:  # only if longer than interval
        while time_current < final_time:
            fn(time_current, min(time_current + interval_duration, final_time))
            time_current += interval_duration


def create_pdf_output(
        frame_data: list[FrameData],
        logged_reward_data: list[dict],
        out_file: str,
        interval_plots: bool = False
):
    """ Create a pdf with all the relevant plots """
    n_frames = len(frame_data)
    times = np.array([frame.time for frame in frame_data])
    frame_ind = np.arange(n_frames)
    frame0 = frame_data[0]
    with (PdfPages(out_file) as pdf):
        # COM position
        com_data = np.array([frame.kinetic_data.com for frame in frame_data])

        # Rewards plot: these are logged separately
        reward_keys = list(logged_reward_data[0].keys())
        reward_data = []
        for rd in logged_reward_data:
            reward_data.append([rd[k] for k in reward_keys])
        reward_data = np.array(reward_data)
        reward_data = np.array(reward_data)
        reward_frame_ind = np.arange(reward_data.shape[0])
        reward_frame_time = np.linspace(0.0, times[-1], reward_data.shape[0])

        rewards_plot = SequencePlot(
            PlotConfig(
                num_vertical=1,
                num_horizontal=1,
                fig_size=(8.5, 6),
                title="Rewards",
                x_label="Time (s)",
                x_label_sub="Env Step",
                y_label="Reward",
                x_data=reward_frame_time,
                x_data_sub=reward_frame_ind,
                x_fmt=".2f",
                x_sub_fmt=".0f",
                y_fmt=".2f",
            )
        )
        for i, key in enumerate(reward_keys):
            if np.all(reward_data[:, i] == 0.0):
                continue
            rewards_plot.add(0, reward_data[:, i], label=key, alpha=0.5)
        rewards_plot.add(0, np.sum(reward_data, axis=1), label="Total")
        rewards_plot.add_hline(0, 0.0)
        rewards_plot.finish(pdf)

        # --- GROUND REACTION FORCE ---
        raw_grf_data = np.array([frame.kinetic_data.grf for frame in frame_data])

        # Express in terms of body weight
        kinetic_data = frame0.kinetic_data
        mass = kinetic_data.total_mass
        weight = abs(float(mass * kinetic_data.gravity))
        grf_bw_data = raw_grf_data / weight

        def create_grf_plot(time_start: float = 0.0, time_end: float = None):
            # Select time range
            if time_end is None:
                time_end = times[-1]
            time_start = max(time_start, times[0])
            time_end = min(time_end, times[-1])
            time_mask = (times >= time_start) & (times <= time_end)
            time_mask = time_mask.flatten()
            time_selected = times[time_mask]
            frame_ind_selected = frame_ind[time_mask]
            grf_selected = grf_bw_data[time_mask, :]

            title = f"Ground Reaction Forces ({time_start:.1f}s to {time_end:.1f}s)"
            grf_plot = SequencePlot(
                PlotConfig(
                    num_vertical=1,
                    num_horizontal=1,
                    fig_size=(8.5, 6),
                    title=title,
                    x_label="Time (s)",
                    x_label_sub="Frame",
                    y_label="GRF (BW)",
                    x_data=time_selected,
                    x_data_sub=frame_ind_selected,
                    x_fmt=".2f",
                    x_sub_fmt=".0f",
                    y_fmt=".1f",
                )
            )
            grf_plot.add_hline(0, 0.0)
            grf_plot.add(0, grf_selected[:, FWD_IDX], label="X")
            grf_plot.add(0, grf_selected[:, UP_IDX], label="Y")
            grf_plot.add(0, grf_selected[:, SIDE_IDX], label="Z")
            grf_plot.finish(pdf)

        # GRF plot for entire duration
        create_grf_plot()

        # Find the intervals in which there is contact
        contact_intervals = []
        contact_threshold = 0.01  # 1% body weight
        in_contact = False
        contact_start = 0.0
        dt = times[1] - times[0]
        for i in range(n_frames):
            # grf_magnitude = np.linalg.norm(grf_data[i, :])
            grf_magnitude = np.abs(grf_bw_data[i, UP_IDX])  # vertical component only
            if not in_contact and grf_magnitude >= contact_threshold:  # start of contact
                in_contact = True
                contact_start = times[i]
            elif in_contact and grf_magnitude < contact_threshold:  # end of contact
                in_contact = False
                contact_end = times[i]
                contact_intervals.append((contact_start - dt, contact_end + dt))

        contact_intervals = np.array(contact_intervals)
        if contact_intervals.size > 0:
            contact_durations = contact_intervals[:, 1] - contact_intervals[:, 0]
            contact_mid_times = 0.5 * (contact_intervals[:, 0] + contact_intervals[:, 1])
            contact_time_plot = SequencePlot(
                PlotConfig(
                    num_vertical=1,
                    num_horizontal=1,
                    fig_size=(8.5, 6),
                    title="Duration of Contacts",
                    x_label="Time (s)",
                    x_label_sub="Frame",
                    y_label="Contact Duration",
                    x_data=times,
                    x_data_sub=frame_ind,
                    x_fmt=".2f",
                    x_sub_fmt=".0f",
                    y_fmt=".3f",
                )
            )
            contact_time_plot.add_scatter(0, contact_mid_times, contact_durations, label="Contact Duration",
                                          connect_line=True, labeled=True)
            contact_time_plot.finish(pdf)

        # --- Step analytics ---
        if contact_intervals.size > 2:
            contact_times = (contact_intervals[:, 0] + contact_intervals[:, 1]) * 0.5
            time_between_steps = []
            length_between_steps = []
            for i in range(1, contact_times.shape[0]):
                t0, t1 = contact_times[i - 1], contact_times[i]
                time_between_steps.append(t1 - t0)
                # Find the frames corresponding to these contact times, grab COM x position
                frame_0, frame_1 = np.searchsorted(times, t0), np.searchsorted(times, t1)
                x0, x1 = com_data[frame_0, 0], com_data[frame_1, 0]
                length_between_steps.append(x1 - x0)

            # Time between steps plot
            time_between_steps = np.array(time_between_steps)
            step_time_plot = SequencePlot(
                PlotConfig(
                    num_vertical=1,
                    num_horizontal=1,
                    fig_size=(8.5, 6),
                    title="Time between Steps",
                    x_label="Time (s)",
                    x_label_sub="Frame",
                    y_label="Time between Steps (s)",
                    x_data=times,
                    x_data_sub=frame_ind,
                    x_fmt=".2f",
                    x_sub_fmt=".0f",
                    y_fmt=".3f",
                )
            )
            step_time_plot.add_scatter(0, contact_times[1:], time_between_steps, label="Time between Steps",
                                       connect_line=True, labeled=True)

            time_between_steps_mean = np.mean(time_between_steps)
            time_between_steps_std = np.std(time_between_steps)
            step_time_plot.add_hline_with_bars(
                0, time_between_steps_mean, time_between_steps_std,
                label=rf"$\mu$: {time_between_steps_mean:.3f}s, $\sigma$: {time_between_steps_std:.3f}s")
            step_time_plot.enforce_y_range(0, 0.0, 0.5)
            step_time_plot.finish(pdf)

            # Length between steps plot
            length_between_steps = np.array(length_between_steps)
            step_length_plot = SequencePlot(
                PlotConfig(
                    num_vertical=1,
                    num_horizontal=1,
                    fig_size=(8.5, 6),
                    title="Length between Steps",
                    x_label="Time (s)",
                    x_label_sub="Frame",
                    y_label="Length between Steps (m)",
                    x_data=times,
                    x_data_sub=frame_ind,
                    x_fmt=".2f",
                    x_sub_fmt=".0f",
                    y_fmt=".3f",
                )
            )
            step_length_plot.add_scatter(0, contact_times[1:], length_between_steps, label="Length between Steps",
                                         connect_line=True, labeled=True)
            length_between_steps_mean = np.mean(length_between_steps)
            length_between_steps_std = np.std(length_between_steps)
            step_length_plot.add_hline_with_bars(
                0, length_between_steps_mean, length_between_steps_std,
                label=rf"$\mu$: {length_between_steps_mean:.3f}m, $\sigma$: {length_between_steps_std:.3f}m")
            step_length_plot.enforce_y_range(0, 0.0, 3.0)
            step_length_plot.finish(pdf)

        # Create grf plots for each contact interval
        dt = times[1] - times[0]
        for (start_time, end_time) in contact_intervals:
            create_grf_plot(start_time - dt, end_time + dt)

        # --- JOINT ANGLES ---
        joint_qpos_names = [j.name for j in frame0.joint_angles]
        joint_angle_limits = [j.limits for j in frame0.joint_angles]
        joint_angles = np.array([[j.value for j in frame.joint_angles] for frame in frame_data])

        def create_joint_angles_plot(time_start: float = 0.0, time_end: float = None):
            # Select time range
            if time_end is None:
                time_end = times[-1]
            time_mask = (times >= time_start) & (times <= time_end)
            time_selected = times[time_mask]
            frame_ind_selected = frame_ind[time_mask]
            title = f"Joint Angles ({time_start:.1f}s to {time_end:.1f}s)"
            create_generic_plot(joint_qpos_names, time_selected, frame_ind_selected, joint_angles[time_mask, :],
                                title, "Value (m or rad)", ".3f", pdf, horizontal_lines=joint_angle_limits)

        # Joint angles plot for entire duration, and 1 second intervals
        create_joint_angles_plot()
        if interval_plots:
            _create_interval_plots(1.0, times, create_joint_angles_plot)

        # --- JOINT MOMENTS ---
        joint_dof_names = [j.name for j in frame0.joint_moments]
        joint_moments = []
        for frame in frame_data:
            joint_moments.append(
                [(m.spring, m.muscle, m.muscle_passive, m.actuator, m.limit, m.damping) for m in frame.joint_moments])
        joint_moments = np.array(joint_moments)
        sublabels = ["Spring", "Muscle", "Muscle Passive", "Actuator", "Limit", "Damping"]

        def create_joint_moments_plot(time_start: float = 0.0, time_end: float = None):
            # Select time range
            if time_end is None:
                time_end = times[-1]
            time_mask = (times >= time_start) & (times <= time_end)
            time_selected = times[time_mask]
            frame_ind_selected = frame_ind[time_mask]
            title = f"Joint Moments ({time_start:.1f}s to {time_end:.1f}s)"
            lss, lsd = "solid", "dashed"
            create_generic_plot(
                names=joint_dof_names,
                times=time_selected,
                frame_ind=frame_ind_selected,
                plot_data=joint_moments[time_mask, :],
                fig_title=title,
                y_label="Value (N m)",
                y_fmt=".1f",
                pdf=pdf,
                sublabels=sublabels,
                alphas=[0.5] * (len(sublabels)),
                linestyles=[lss] * (len(sublabels)),
                horizontal_lines=[[0.0]] * len(joint_dof_names),
                omit_zeros=True
            )

        # Joint moments plot for entire duration, and 1 second intervals
        create_joint_moments_plot()
        if interval_plots:
            _create_interval_plots(1.0, times, create_joint_moments_plot)

        # --- JOINT PASSIVE MUSCLE FORCE ANALYSIS ---
        muscle_names = [m.name for m in frame0.muscles]
        qpos_values = np.array([[j.value for j in frame.joint_angles] for frame in frame_data])

        def create_joint_breakdown_plot(
                title: str,
                joint_muscle_forces: np.ndarray,
                time_start: float = 0.0,
                time_end: float = None
        ):
            # Select time range
            if time_end is None:
                time_end = times[-1]
            time_mask = (times >= time_start) & (times <= time_end)
            time_selected = times[time_mask]
            frame_ind_selected = frame_ind[time_mask]

            title = f"{title} ({time_start:.1f}s to {time_end:.1f}s)"
            lss, lsd = "solid", "dashed"
            create_generic_plot(
                names=joint_qpos_names,
                times=time_selected,
                frame_ind=frame_ind_selected,
                plot_data=joint_muscle_forces[time_mask, :],
                fig_title=title,
                y_label="Value (N m)",
                y_fmt=".1f",
                pdf=pdf,
                sublabels=muscle_names,
                alphas=[0.5] * (len(muscle_names)),
                linestyles=[lss] * (len(muscle_names)),
                horizontal_lines=[[0.0]] * len(joint_qpos_names),
                omit_zeros=True,
                secondary_overlay_data=qpos_values.T,
            )

        joint_passive_muscle, joint_active_muscle = [], []
        for frame in frame_data:
            joint_passive_muscle.append([m.passive_breakdown for m in frame.joint_muscle_breakdown])
            joint_active_muscle.append([m.active_breakdown for m in frame.joint_muscle_breakdown])
        joint_passive_muscle = np.array(joint_passive_muscle)  # [n_frames, n_qpos, n_muscles]
        joint_active_muscle = np.array(joint_active_muscle)

        create_joint_breakdown_plot(title="Joint Passive Breakdown", joint_muscle_forces=joint_passive_muscle)
        create_joint_breakdown_plot(title="Joint Active Breakdown", joint_muscle_forces=joint_active_muscle)

        # --- BODY FORCES ---
        # body_names = [b.name for b in frame0.body_forces]
        # body_forces = []
        # for frame in frame_data:
        #     body_forces.append(
        #         [(b.net_force, b.gravity, b.contact, b.muscle, b.drag) for b in frame.body_forces])
        # body_forces = np.array(body_forces)
        # body_forces = body_forces[..., 0] # test
        # sublabels = ["Net", "Gravity", "Contact", "Muscle", "Drag"]
        #
        # def create_body_forces_plot(time_start: float = 0.0, time_end: float = None):
        #     if time_end is None:
        #         time_end = times[-1]
        #     time_mask = (times >= time_start) & (times <= time_end)
        #     time_selected = times[time_mask]
        #     frame_ind_selected = frame_ind[time_mask]
        #     title = f"Body Forces ({time_start:.1f}s to {time_end:.1f}s)"
        #     lss, lsd = "solid", "dashed"
        #     create_generic_plot(
        #         names=body_names,
        #         times=time_selected,
        #         frame_ind=frame_ind_selected,
        #         plot_data=body_forces[time_mask, :],
        #         fig_title=title,
        #         y_label="Value (N m)",
        #         y_fmt=".1f",
        #         pdf=pdf,
        #         sublabels=sublabels,
        #         alphas=[0.5] * (len(sublabels)),
        #         linestyles=[lss] * (len(sublabels)),
        #         horizontal_lines=[[0.0]] * len(joint_dof_names),
        #         omit_zeros=True
        #     )
        # create_body_forces_plot()

        # --- MUSCLE PLOTS ---
        # Muscle activations, fiber/tendon lengths, moment arms
        muscle_ae = []
        muscle_ftl = []
        muscle_ma = []
        muscle_frc = []
        muscle_frc_mult = []
        for frame in frame_data:
            muscle_ae.append([(m.activation, m.excitation) for m in frame.muscles])
            muscle_ftl.append([(
                m.fiber_length / m.optimal_fiber_length,
                # m.tendon_length / m.tendon_slack_length,
                bolt.MIN_NORM_FIBER_LENGTH,
                bolt.MAX_NORM_FIBER_LENGTH,
            ) for m in frame.muscles])
            muscle_ma.append([m.moment_arm for m in frame.muscles])
            muscle_frc.append([(m.actuation, m.max_isometric_force) for m in frame.muscles])
            muscle_frc_mult.append([
                (m.length_multiplier, m.velocity_multiplier, m.passive_multiplier, m.damping_multiplier)
                for m in frame.muscles])

        muscle_ae = np.array(muscle_ae)
        muscle_ftl = np.array(muscle_ftl)
        muscle_ma = np.array(muscle_ma)
        muscle_frc = np.array(muscle_frc)
        muscle_frc_mult = np.array(muscle_frc_mult)
        zero_lines = [[0.0, 0.0]] * len(muscle_names)

        # Activation/excitation
        create_generic_plot(muscle_names, times, frame_ind, muscle_ae,
                            "Muscle Activations/Excitations", "Activation/Excitation", ".2f",
                            pdf, enforced_y_range=[(0.0, 1.0)] * len(muscle_names),
                            sublabels=["Activation", "Excitation"],
                            alphas=[1.0, 0.5], horizontal_lines=zero_lines)

        # Fiber/tendon lengths
        enforced_range = []
        for m in frame0.muscles:
            min_range = bolt.MIN_NORM_FIBER_LENGTH
            max_range = bolt.MAX_NORM_FIBER_LENGTH
            enforced_range.append((min_range, max_range))
        create_generic_plot(
            muscle_names, times, frame_ind, muscle_ftl,
            "Muscle Fiber/Tendon Length", "Length (m)", ".3f",
            pdf,
            sublabels=["Norm Fiber", "Min Fiber", "Max Fiber"],
            alphas=[1.0, 0.5, 0.5],
            linestyles=["solid", "dashed", "dashed"],
            enforced_y_range=enforced_range)

        # Moment arms
        enforced_range = [(-0.1, 0.1)] * len(muscle_names)
        create_generic_plot(muscle_names, times, frame_ind, muscle_ma,
                            "Muscle Moment Arms", "Moment Arm (m)", ".2f",
                            pdf, sublabels=joint_qpos_names, omit_zeros=True, omit_tolerance=1e-3,
                            horizontal_lines=zero_lines, enforced_y_range=enforced_range)

        # Muscle actuation
        enforced_range = [(0.0, 2.0 * m.max_isometric_force) for m in frame0.muscles]
        create_generic_plot(muscle_names, times, frame_ind, muscle_frc,
                            "Muscle Actuation", "Actuation (N)", ".1f",
                            pdf, sublabels=["Actuation", "Max Isometric Force"],
                            alphas=[1.0, 0.5],
                            enforced_y_range=enforced_range)

        # Muscle force multipliers
        create_generic_plot(muscle_names, times, frame_ind, muscle_frc_mult,
                            "Muscle Force Multipliers", "Multiplier", ".2f",
                            pdf, sublabels=["Active Length", "Active Velocity", "Passive Length", "Damping"],
                            alphas=[0.5] * muscle_frc_mult.shape[-1],
                            enforced_y_range=[(0.0, 2.5)] * len(muscle_names))

        # --- ACTUATOR PLOTS ---
        actuator_names = [a.name for a in frame0.actuators]
        actuator_ae = []
        for frame in frame_data:
            actuator_ae.append([(a.activation, a.excitation) for a in frame.actuators])
        actuator_ae = np.array(actuator_ae)
        create_generic_plot(actuator_names, times, frame_ind, np.array(actuator_ae),
                            "Actuator Activations/Excitations", "Activation/Excitation", ".2f",
                            pdf, enforced_y_range=[(0.0, 1.0)] * len(actuator_names),
                            sublabels=["Activation", "Excitation"],
                            alphas=[1.0, 0.5],
                            horizontal_lines=[[0.5]] * len(actuator_names))
