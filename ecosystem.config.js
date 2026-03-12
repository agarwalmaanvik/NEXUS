module.exports = {
    apps: [
        {
            name: "nexus-backend",
            script: "nexus_server.py",
            interpreter: "python3",
            watch: false,
            env: {
                PORT: 8000,
                NODE_ENV: "production"
            }
        },
        {
            name: "nexus-frontend",
            script: "npm",
            args: "run dev -- --host",
            cwd: "./web_gui",
            watch: false
        }
    ]
};
