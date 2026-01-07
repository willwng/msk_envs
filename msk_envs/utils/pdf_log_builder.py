import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from msk_envs.utils.checkpoint_parser import FrameData
from .plot_helper import SequencePlot, PlotConfig


def create_generic_plot(
        names: list[str],
        times: np.ndarray,
        frame_ind: np.ndarray,
        plot_data: np.ndarray,
        fig_title: str,
        y_label: str,
        y_fmt: str,
        pdf: PdfPages,
        enforced_range: tuple[float, float] = None,
        sublabels: list[str] = None,
        alphas: list[float] = None,
        horizontal_lines: list[list[float]] = None,
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
                        seq_plot.add(idx_fig, data_subset[..., part],
                                     label=label[part],
                                     alpha=alpha,
                                     title=title)

                # Add horizontal lines if specified. can be used for zero lines or joint limits
                idx_entry = start_idx + i
                if horizontal_lines and horizontal_lines[idx_entry] is not None:
                    for hline in horizontal_lines[idx_entry]:
                        seq_plot.add_hline(idx_fig, hline)

            # Enforce y range if specified
            if enforced_range is not None:
                seq_plot.enforce_y_range(idx_fig, enforced_range[0], enforced_range[1])

        seq_plot.finish(pdf)


def create_interval_plots(interval_duration: float, times: np.ndarray, fn):
    time_current, final_time = 0.0, times[-1]
    if final_time - time_current > interval_duration:  # only if longer than interval
        while time_current < final_time:
            fn(time_current, min(time_current + interval_duration, final_time))
            time_current += interval_duration
    return


def create_pdf_output(frame_data: list[FrameData], out_file: str):
    """ Create a pdf with all the relevant plots """
    n_frames = len(frame_data)
    times = np.array([frame.time for frame in frame_data])
    frame_ind = np.arange(n_frames)
    with (PdfPages(out_file) as pdf):
        # Rewards plot
        reward_keys = list(frame_data[0].reward_data.keys())
        reward_data = []
        for frame in frame_data:
            reward_data.append([frame.reward_data[k] for k in reward_keys])
        reward_data = np.array(reward_data)

        rewards_plot = SequencePlot(
            PlotConfig(
                num_vertical=1,
                num_horizontal=1,
                fig_size=(8.5, 6),
                title="Rewards",
                x_label="Time (s)",
                x_label_sub="Frame",
                y_label="Reward",
                x_data=times,
                x_data_sub=frame_ind,
                x_fmt=".2f",
                x_sub_fmt=".0f",
                y_fmt=".1f",
            )
        )
        for i, key in enumerate(reward_keys):
            rewards_plot.add(0, reward_data[:, i], label=key)
        rewards_plot.add(0, np.sum(reward_data, axis=1), label="Total")
        rewards_plot.add_hline(0, 0.0)
        rewards_plot.finish(pdf)

        # --- GROUND REACTION FORCE ---
        grf_data = np.array([frame.kinetic_data.grf for frame in frame_data])

        # Express in terms of body weight
        kinetic_data = frame_data[0].kinetic_data
        mass = kinetic_data.total_mass
        weight = abs(float(mass * kinetic_data.gravity))
        grf_data /= weight

        def create_grf_plot(time_start: float = 0.0, time_end: float = None):
            # Select time range
            if time_end is None:
                time_end = times[-1]
            time_mask = (times >= time_start) & (times <= time_end)
            time_mask = time_mask.flatten()
            time_selected = times[time_mask]
            frame_ind_selected = frame_ind[time_mask]
            grf_selected = grf_data[time_mask, :]

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

            # put grf_selected through a low pass filter to reduce noise
            # from scipy.signal import butter, filtfilt
            # def lowpass(data, times, order=4):
            #     fs = 1.0 / np.mean(np.diff(times))  # sampling frequency (Hz)
            #     cutoff = 60.0  # Hz
            #
            #     nyq = 0.5 * fs
            #     normal_cutoff = cutoff / nyq
            #
            #     b, a = butter(order, normal_cutoff, btype='low')
            #     return filtfilt(b, a, data, axis=0)
            # grf_selected = lowpass(grf_selected, time_selected)

            grf_plot.add_hline(0, 0.0)
            grf_plot.add(0, grf_selected[:, 0], label="X")
            grf_plot.add(0, grf_selected[:, 1], label="Y")
            grf_plot.add(0, grf_selected[:, 2], label="Z")

            # Compute the impulse over the selected time range
            impulse = np.trapz(grf_selected, time_selected, axis=0)
            # grf_plot.add_text(0, x_pos=0.25, y_pos=3.15,
            #                   text=f"Impulse (BW s): ({impulse[0]:.2f}, {impulse[1]:.2f}, {impulse[2]:.2f})",
            #                   fontsize=6)
            # print(f"GRF Impulse (BW s) from {time_start:.4f}s to {time_end:.4f}s: "
            #       f"({impulse[0]:.4f}, {impulse[1]:.4f}, {impulse[2]:.4f})")

            grf_plot.finish(pdf)

        # GRF plot for entire duration
        create_grf_plot()

        # Find the intervals in which there is contact
        contact_intervals = []
        contact_threshold = 0.01  # 5% of body weight
        in_contact = False
        contact_start = 0.0
        for i in range(n_frames):
            grf_magnitude = np.linalg.norm(grf_data[i, :])
            if not in_contact and grf_magnitude >= contact_threshold:
                in_contact = True
                contact_start = times[i]
            elif in_contact and grf_magnitude < contact_threshold:
                in_contact = False
                contact_end = times[i]
                contact_intervals.append((contact_start, contact_end))
        if contact_intervals:
            contact_intervals = np.array(contact_intervals)
            contact_durations = contact_intervals[:, 1] - contact_intervals[:, 0]
            contact_mid_times = 0.5 * (contact_intervals[:, 0] + contact_intervals[:, 1])
            contact_time_plot = SequencePlot(
                PlotConfig(
                    num_vertical=1,
                    num_horizontal=1,
                    fig_size=(8.5, 6),
                    title="Contact Durations",
                    x_label="Time (s)",
                    x_label_sub="Frame",
                    y_label="Contact Duration",
                    x_data=times,
                    x_data_sub=frame_ind,
                    x_fmt=".2f",
                    x_sub_fmt=".0f",
                    y_fmt=".2f",
                )
            )
            contact_time_plot.add_scatter(0, contact_mid_times, contact_durations, label="Contact Duration",
                                          connect_line=True, labeled=True)
            contact_time_plot.finish(pdf)

        # Create interval plots for each contact interval
        for (start_time, end_time) in contact_intervals:
            create_grf_plot(start_time, end_time)

        # --- JOINT ANGLES ---
        has_reference = frame_data[0].joint_angles[0].has_reference()
        joint_names = [j.name for j in frame_data[0].joint_angles]
        joint_angles = []
        joint_angle_limits = [j.limits for j in frame_data[0].joint_angles]
        for frame in frame_data:
            if has_reference:
                joint_angles.append([(j.value, j.reference) for j in frame.joint_angles])
            else:
                joint_angles.append([j.value for j in frame.joint_angles])
        joint_angles = np.array(joint_angles)

        def create_joint_angles_plot(time_start: float = 0.0, time_end: float = None):
            # Select time range
            if time_end is None:
                time_end = times[-1]
            time_mask = (times >= time_start) & (times <= time_end)
            time_selected = times[time_mask]
            frame_ind_selected = frame_ind[time_mask]
            title = f"Joint Angles ({time_start:.1f}s to {time_end:.1f}s)"
            sublabels = ["Value", "Reference"] if has_reference else None
            alpha = [1.0, 0.5] if has_reference else None
            create_generic_plot(joint_names, time_selected, frame_ind_selected, joint_angles[time_mask, :],
                                title, "Value (m or rad)", ".3f", pdf, sublabels=sublabels, alphas=alpha,
                                horizontal_lines=joint_angle_limits)

        # Joint angles plot for entire duration, and 1 second intervals
        create_joint_angles_plot()
        create_interval_plots(1.0, times, create_joint_angles_plot)

        # --- JOINT MOMENTS ---
        joint_names = [j.name for j in frame_data[0].joint_moments]
        joint_moments = []
        for frame in frame_data:
            joint_moments.append([j.value for j in frame.joint_moments])
        joint_moments = np.array(joint_moments)

        def create_joint_moments_plot(time_start: float = 0.0, time_end: float = None):
            # Select time range
            if time_end is None:
                time_end = times[-1]
            time_mask = (times >= time_start) & (times <= time_end)
            time_selected = times[time_mask]
            frame_ind_selected = frame_ind[time_mask]
            title = f"Joint Moments ({time_start:.1f}s to {time_end:.1f}s)"
            create_generic_plot(joint_names, time_selected, frame_ind_selected, joint_moments[time_mask, :],
                                title, "Value (N m)", ".3f", pdf)

        # Joint moments plot for entire duration, and 1 second intervals
        create_joint_moments_plot()
        create_interval_plots(1.0, times, create_joint_moments_plot)

        # --- MUSCLE PLOTS ---
        muscle_names = [m.name for m in frame_data[0].muscles]
        # Muscle activations, fiber/tendon lengths
        muscle_ae = []
        muscle_ftl = []
        for frame in frame_data:
            muscle_ae.append([(m.activation, m.excitation) for m in frame.muscles])
            muscle_ftl.append([(m.fiber_length, m.tendon_length) for m in frame.muscles])

        muscle_ftl = np.array(muscle_ftl)
        zero_lines = [[0.0, 0.0]] * len(muscle_names)
        create_generic_plot(muscle_names, times, frame_ind, np.array(muscle_ae),
                            "Muscle Activations/Excitations", "Activation/Excitation", ".2f",
                            pdf, enforced_range=(0.0, 1.0),
                            sublabels=["Activation", "Excitation"],
                            alphas=[1.0, 0.5], horizontal_lines=zero_lines)
        create_generic_plot(muscle_names, times, frame_ind, np.array(muscle_ftl),
                            "Muscle Fiber/Tendon Length", "Length (m)", ".3f",
                            pdf, sublabels=["Fiber", "Tendon"], horizontal_lines=zero_lines)

        # --- ACTUATOR PLOTS ---
        actuator_names = [a.name for a in frame_data[0].actuators]
        actuator_ae = []
        for frame in frame_data:
            actuator_ae.append([(a.activation, a.excitation) for a in frame.actuators])
        actuator_ae = np.array(actuator_ae)
        create_generic_plot(actuator_names, times, frame_ind, np.array(actuator_ae),
                            "Actuator Activations/Excitations", "Activation/Excitation", ".2f",
                            pdf, enforced_range=(0.0, 1.0),
                            sublabels=["Activation", "Excitation"],
                            alphas=[1.0, 0.5])

    return
