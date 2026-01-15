"""Test bulk save, dump, and load functionality for multiple fields."""

import os
import tempfile

import pytest

try:
    import sampai as sam
except ImportError:
    pytest.skip("sampai module not built", allow_module_level=True)


class TestBulkSave:
    """Tests for bulk save() function with arbitrary number of fields."""

    def test_bulk_save_2d_single_field(self):
        """Test bulk save with a single 2D field (backward compatibility)."""
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 4

        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.make(box, config)

        # Create a field
        field = sam.field.scalar(mesh, "u", init=1.5)

        # Save with bulk API
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_single")
            sam.save(filepath, field)

            # Verify files were created
            h5_file = os.path.join(tmpdir, "test_single.h5")
            xdmf_file = os.path.join(tmpdir, "test_single.xdmf")
            assert os.path.exists(h5_file)
            assert os.path.exists(xdmf_file)

    def test_bulk_save_2d_multiple_fields(self):
        """Test bulk save with multiple 2D fields."""
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 4

        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.make(box, config)

        # Create multiple fields with different initial values
        rho = sam.field.scalar(mesh, "rho", init=1.0)
        u = sam.field.scalar(mesh, "u", init=0.5)
        v = sam.field.scalar(mesh, "v", init=0.3)
        p = sam.field.scalar(mesh, "p", init=0.8)

        # Save all fields with bulk API
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_multi")
            sam.save(filepath, rho, u, v, p)

            # Verify files were created
            h5_file = os.path.join(tmpdir, "test_multi.h5")
            xdmf_file = os.path.join(tmpdir, "test_multi.xdmf")
            assert os.path.exists(h5_file)
            assert os.path.exists(xdmf_file)

    def test_bulk_save_2d_many_fields(self):
        """Test bulk save with up to 8 2D fields."""
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 4

        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.make(box, config)

        # Create 8 fields
        fields = []
        for i in range(8):
            field = sam.field.scalar(mesh, f"field_{i}", init=float(i))
            fields.append(field)

        # Save all fields with bulk API
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_many")
            sam.save(filepath, *fields)

            # Verify files were created
            h5_file = os.path.join(tmpdir, "test_many.h5")
            xdmf_file = os.path.join(tmpdir, "test_many.xdmf")
            assert os.path.exists(h5_file)
            assert os.path.exists(xdmf_file)

    def test_bulk_save_2d_too_many_fields(self):
        """Test that bulk save raises error for more than 8 fields."""
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 4

        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.make(box, config)

        # Create 9 fields (one too many)
        fields = []
        for i in range(9):
            field = sam.field.scalar(mesh, f"field_{i}", init=float(i))
            fields.append(field)

        # Should raise an error
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_error")
            with pytest.raises(RuntimeError, match="Maximum 8 fields"):
                sam.save(filepath, *fields)

    def test_bulk_save_same_mesh_validation(self):
        """Test that bulk save validates all fields share the same mesh."""
        # Create two different meshes
        box1 = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config1 = sam.config.make(2)
        config1.min_level = 2
        config1.max_level = 4
        mesh1 = sam.mesh.make(box1, config1)

        box2 = sam.geometry.box([0.0, 0.0], [2.0, 2.0])
        config2 = sam.config.make(2)
        config2.min_level = 2
        config2.max_level = 4
        mesh2 = sam.mesh.make(box2, config2)

        # Create fields on different meshes
        field1 = sam.field.scalar(mesh1, "f1", init=0.0)
        field2 = sam.field.scalar(mesh2, "f2", init=0.0)

        # Should raise an error
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_error")
            with pytest.raises(RuntimeError, match="same mesh"):
                sam.save(filepath, field1, field2)

    def test_bulk_save_1d(self):
        """Test bulk save with 1D fields."""
        config = sam.config.make(1)
        config.min_level = 2
        config.max_level = 4

        box = sam.geometry.box([0.0], [1.0])
        mesh = sam.mesh.make(box, config)

        # Create multiple fields
        u = sam.field.scalar(mesh, "u", init=1.0)
        v = sam.field.scalar(mesh, "v", init=2.0)
        w = sam.field.scalar(mesh, "w", init=3.0)

        # Save with bulk API
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_1d")
            sam.save(filepath, u, v, w)

            # Verify files were created
            h5_file = os.path.join(tmpdir, "test_1d.h5")
            xdmf_file = os.path.join(tmpdir, "test_1d.xdmf")
            assert os.path.exists(h5_file)
            assert os.path.exists(xdmf_file)

    def test_bulk_save_3d(self):
        """Test bulk save with 3D fields."""
        config = sam.config.make(3)
        config.min_level = 1
        config.max_level = 2

        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        mesh = sam.mesh.make(box, config)

        # Create multiple fields
        rho = sam.field.scalar(mesh, "rho", init=1.0)
        u = sam.field.scalar(mesh, "u", init=0.5)
        v = sam.field.scalar(mesh, "v", init=0.3)

        # Save with bulk API
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_3d")
            sam.save(filepath, rho, u, v)

            # Verify files were created
            h5_file = os.path.join(tmpdir, "test_3d.h5")
            xdmf_file = os.path.join(tmpdir, "test_3d.xdmf")
            assert os.path.exists(h5_file)
            assert os.path.exists(xdmf_file)


class TestBulkDump:
    """Tests for bulk dump() function with arbitrary number of fields."""

    def test_bulk_dump_2d_single_field(self):
        """Test bulk dump with a single 2D field (backward compatibility)."""
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 4

        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.make(box, config)

        # Create a field
        field = sam.field.scalar(mesh, "u", init=1.5)

        # Dump with bulk API
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_single")
            sam.dump(filepath, field)

            # Verify HDF5 file was created (no XDMF for dump)
            h5_file = os.path.join(tmpdir, "test_single.h5")
            xdmf_file = os.path.join(tmpdir, "test_single.xdmf")
            assert os.path.exists(h5_file)
            assert not os.path.exists(xdmf_file)

    def test_bulk_dump_2d_multiple_fields(self):
        """Test bulk dump with multiple 2D fields."""
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 4

        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.make(box, config)

        # Create multiple fields with different initial values
        rho = sam.field.scalar(mesh, "rho", init=1.0)
        u = sam.field.scalar(mesh, "u", init=0.5)
        v = sam.field.scalar(mesh, "v", init=0.3)
        p = sam.field.scalar(mesh, "p", init=0.8)

        # Dump all fields with bulk API
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_multi")
            sam.dump(filepath, rho, u, v, p)

            # Verify HDF5 file was created (no XDMF for dump)
            h5_file = os.path.join(tmpdir, "test_multi.h5")
            assert os.path.exists(h5_file)

    def test_bulk_dump_2d_many_fields(self):
        """Test bulk dump with up to 8 2D fields."""
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 4

        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.make(box, config)

        # Create 8 fields
        fields = []
        for i in range(8):
            field = sam.field.scalar(mesh, f"field_{i}", init=float(i))
            fields.append(field)

        # Dump all fields with bulk API
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_many")
            sam.dump(filepath, *fields)

            # Verify HDF5 file was created
            h5_file = os.path.join(tmpdir, "test_many.h5")
            assert os.path.exists(h5_file)

    def test_bulk_dump_1d(self):
        """Test bulk dump with 1D fields."""
        config = sam.config.make(1)
        config.min_level = 2
        config.max_level = 4

        box = sam.geometry.box([0.0], [1.0])
        mesh = sam.mesh.make(box, config)

        # Create multiple fields
        u = sam.field.scalar(mesh, "u", init=1.0)
        v = sam.field.scalar(mesh, "v", init=2.0)
        w = sam.field.scalar(mesh, "w", init=3.0)

        # Dump with bulk API
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_1d")
            sam.dump(filepath, u, v, w)

            # Verify HDF5 file was created
            h5_file = os.path.join(tmpdir, "test_1d.h5")
            assert os.path.exists(h5_file)

    def test_bulk_dump_3d(self):
        """Test bulk dump with 3D fields."""
        config = sam.config.make(3)
        config.min_level = 1
        config.max_level = 2

        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        mesh = sam.mesh.make(box, config)

        # Create multiple fields
        rho = sam.field.scalar(mesh, "rho", init=1.0)
        u = sam.field.scalar(mesh, "u", init=0.5)
        v = sam.field.scalar(mesh, "v", init=0.3)

        # Dump with bulk API
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_3d")
            sam.dump(filepath, rho, u, v)

            # Verify HDF5 file was created
            h5_file = os.path.join(tmpdir, "test_3d.h5")
            assert os.path.exists(h5_file)


class TestBulkLoad:
    """Tests for bulk load() function with arbitrary number of fields."""

    def test_bulk_dump_load_2d_multiple_fields(self):
        """Test bulk dump and load with multiple 2D fields."""
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 4

        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sam.mesh.make(box, config)

        # Create multiple fields with specific initial values
        rho = sam.field.scalar(mesh, "rho", init=1.0)
        u = sam.field.scalar(mesh, "u", init=0.5)
        v = sam.field.scalar(mesh, "v", init=0.3)
        p = sam.field.scalar(mesh, "p", init=0.8)

        # Dump all fields with bulk API
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_restart")
            sam.dump(filepath, rho, u, v, p)

            # Create new mesh and fields for loading
            mesh2 = sam.mesh.make(box, config)
            rho2 = sam.field.scalar(mesh2, "rho", init=0.0)
            u2 = sam.field.scalar(mesh2, "u", init=0.0)
            v2 = sam.field.scalar(mesh2, "v", init=0.0)
            p2 = sam.field.scalar(mesh2, "p", init=0.0)

            # Load all fields with bulk API
            sam.load(filepath, rho2, u2, v2, p2)

            # Verify the file still exists
            h5_file = os.path.join(tmpdir, "test_restart.h5")
            assert os.path.exists(h5_file)

    def test_bulk_load_1d_multiple_fields(self):
        """Test bulk dump and load with multiple 1D fields."""
        config = sam.config.make(1)
        config.min_level = 2
        config.max_level = 4

        box = sam.geometry.box([0.0], [1.0])
        mesh = sam.mesh.make(box, config)

        # Create multiple fields
        u = sam.field.scalar(mesh, "u", init=1.0)
        v = sam.field.scalar(mesh, "v", init=2.0)
        w = sam.field.scalar(mesh, "w", init=3.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_1d_restart")
            sam.dump(filepath, u, v, w)

            # Create new mesh and fields for loading
            mesh2 = sam.mesh.make(box, config)
            u2 = sam.field.scalar(mesh2, "u", init=0.0)
            v2 = sam.field.scalar(mesh2, "v", init=0.0)
            w2 = sam.field.scalar(mesh2, "w", init=0.0)

            # Load all fields with bulk API
            sam.load(filepath, u2, v2, w2)

            h5_file = os.path.join(tmpdir, "test_1d_restart.h5")
            assert os.path.exists(h5_file)

    def test_bulk_load_3d_multiple_fields(self):
        """Test bulk dump and load with multiple 3D fields."""
        config = sam.config.make(3)
        config.min_level = 1
        config.max_level = 2

        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        mesh = sam.mesh.make(box, config)

        # Create multiple fields
        rho = sam.field.scalar(mesh, "rho", init=1.0)
        u = sam.field.scalar(mesh, "u", init=0.5)
        v = sam.field.scalar(mesh, "v", init=0.3)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_3d_restart")
            sam.dump(filepath, rho, u, v)

            # Create new mesh and fields for loading
            mesh2 = sam.mesh.make(box, config)
            rho2 = sam.field.scalar(mesh2, "rho", init=0.0)
            u2 = sam.field.scalar(mesh2, "u", init=0.0)
            v2 = sam.field.scalar(mesh2, "v", init=0.0)

            # Load all fields with bulk API
            sam.load(filepath, rho2, u2, v2)

            h5_file = os.path.join(tmpdir, "test_3d_restart.h5")
            assert os.path.exists(h5_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
