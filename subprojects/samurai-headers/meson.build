# Samurai Headers - Meson Subproject
#
# This subproject manages downloading and exposing the Samurai C++ headers.
# Samurai is a header-only library that doesn't have a Meson build system,
# so we provide a thin wrapper to fetch it and expose the headers.
#
# Usage (from parent project):
#   samurai_subproj = subproject('samurai-headers')
#   samurai_dep = samurai_subproj.get_variable('samurai_dep')
#
# Options:
#   -Dsamurai_commit=<commit|tag|branch>  # Which version to fetch (default: main)
#   -Dsamurai_depth=<0|1>                 # Clone depth (default: 1 = shallow)
#   -Dsamurai_local_path=<path>           # Use local copy instead of fetching

project('samurai-headers',
    'cpp',
    version: '0.27.1',
    meson_version: '>=0.64.0',
    default_options: [
        'cpp_std=c++20',
    ]
)

# ============================================================================
# OPTIONS
# ============================================================================

# Get options (passed from parent project or use defaults)
samurai_commit = get_option('samurai_commit')
samurai_depth = get_option('samurai_depth')
samurai_local_path = get_option('samurai_local_path')

# Samurai repository info
samurai_repo_url = 'https://github.com/hpc-maths/samurai.git'
samurai_download_dir = meson.current_source_dir() / 'samurai'

message('')
message('=== Samurai Headers Subproject ===')

# ============================================================================
# LOCAL PATH OR DOWNLOAD FROM GITHUB
# ============================================================================

if samurai_local_path != ''
    # Use local Samurai installation
    message('Using local Samurai: @0@'.format(samurai_local_path))

    local_check = run_command('test', '-d', samurai_local_path / 'include' / 'samurai', check: false)
    if local_check.returncode() != 0
        error('Samurai not found at local path: @0@'.format(samurai_local_path))
    endif

    samurai_include = include_directories(samurai_local_path / 'include')

else
    # Download from GitHub
    message('Fetching Samurai from GitHub...')
    message('  Repository: @0@'.format(samurai_repo_url))
    message('  Revision: @0@'.format(samurai_commit))
    message('  Depth: @0@'.format(samurai_depth == 0 ? 'full' : 'shallow (@0@)'.format(samurai_depth)))

    # Check if already downloaded at correct commit
    samurai_git_head = samurai_download_dir / '.git' / 'HEAD'
    need_download = true

    if run_command('test', '-f', samurai_git_head, check: false).returncode() == 0
        # Get current commit
        get_commit = run_command('git', '-C', samurai_download_dir, 'rev-parse', 'HEAD', check: false)
        if get_commit.returncode() == 0
            current_commit = get_commit.stdout().strip()

            # Check if already at requested commit/branch
            if current_commit.startswith(samurai_commit) or samurai_commit == 'main'
                message('  Samurai already at requested commit')
                need_download = false
            else
                message('  Switching from @0@ to @1@...'.format(current_commit.substring(0, 8), samurai_commit))

                # Try to fetch and switch
                fetch_cmd = run_command('git', '-C', samurai_download_dir,
                                       'fetch', '--depth=' + samurai_depth.to_string(),
                                       'origin', samurai_commit, check: false)
                if fetch_cmd.returncode() == 0
                    checkout_cmd = run_command('git', '-C', samurai_download_dir,
                                              'checkout', samurai_commit, check: false)
                    if checkout_cmd.returncode() == 0
                        need_download = false
                        message('  Successfully switched!')
                    endif
                endif
            endif
        endif
    endif

    # Download if needed
    if need_download
        message('  Downloading Samurai...')

        # Remove existing directory
        run_command('rm', '-rf', samurai_download_dir, check: false)

        # Clone with git
        clone_args = ['git', 'clone', '--single-branch', '--quiet']

        if samurai_depth > 0
            clone_args += ['--depth=' + samurai_depth.to_string()]
        endif

        clone_args += ['--branch=' + samurai_commit]
        clone_args += [samurai_repo_url, samurai_download_dir]

        clone_cmd = run_command(clone_args, check: true)
        message('  Download complete!')
    endif

    samurai_include = include_directories('samurai/include', is_system: false)
endif

# ============================================================================
# DECLARE DEPENDENCY FOR PARENT PROJECT
# ============================================================================

samurai_dep = declare_dependency(
    include_directories: samurai_include,
    dependencies: [],  # Dependencies provided by parent project
)

message('  Headers available at: include/samurai')
message('=====================================')
message('')
