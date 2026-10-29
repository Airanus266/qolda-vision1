# Data and dependency boundaries

The application does not intentionally upload camera frames or collected features. MediaPipe assets are loaded from jsDelivr and Google; fonts are requested from Google Fonts. Those hosts receive ordinary network request metadata.

Imported JSON is treated as untrusted, size-limited and schema-validated. Supported class names are fixed; imported strings are not accepted as HTML. Exports are user-initiated. Data remains in tab memory unless explicitly downloaded.

If you find a vulnerability, contact the repository owner privately through a contact method they publish. Do not post actual participant data or credentials in a public issue. No dedicated security response team or audit is claimed.
