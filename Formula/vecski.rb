class Vecski < Formula
  desc "API server that fits and applies translators between embedding spaces"
  homepage "https://github.com/looskis/vecski"
  url "https://github.com/looskis/vecski/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "56f154f218ab30b5d62bad0553c734a2f01804cf5107ad325aa1b6839e151153"
  license "Apache-2.0"
  head "https://github.com/looskis/vecski.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "cargo", "install", *std_cargo_args(path: "crates/vecski-server")
  end

  service do
    run [opt_bin/"vecski", "--bind", "127.0.0.1:8080", "--data-dir", var/"vecski"]
    keep_alive true
    working_dir var/"vecski"
    log_path var/"log/vecski.log"
    error_log_path var/"log/vecski.log"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/vecski --version")

    port = free_port
    pid = spawn bin/"vecski", "--bind", "127.0.0.1:#{port}", "--data-dir", testpath/"data"
    begin
      sleep 2
      assert_match "\"status\":\"ok\"", shell_output("curl -s http://127.0.0.1:#{port}/healthz")
    ensure
      Process.kill("TERM", pid)
      Process.wait(pid)
    end
  end
end
