# Mobile manipulator workspace

Husky A200 + Kinova Gen3 (7 DoF) + Unitree Dex3-1 hand in Gazebo Harmonic with
MoveIt 2, on ROS 2 Jazzy, inside a dev container.

```bash
make env            # once: write .devcontainer/.env for this host (UID, DISPLAY)
make up             # build the image if needed, start the container, allow X11
make check          # preflight inside the container; starts no processes
make run            # Gazebo + robot + MoveIt + RViz; Ctrl+C stops everything
```

VS Code: open this folder and "Dev Containers: Reopen in Container".
See `CLAUDE.md` for the layout, the aliases and the known pitfalls.
